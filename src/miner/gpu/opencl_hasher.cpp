#include "opencl_hasher.hpp"
#include "crypto/hasher_sha256.hpp"
#include "general/hex.hpp"
#include "helpers.hpp"
#include "kernel.hpp"
#include "spdlog/spdlog.h"
#include <iostream>

// not a right way to place it here, TODO: to make a cleaner code)
extern uint32_t gpubatchsize; 
extern bool autofiltering;
extern double gpu_manual_filter_bound;
extern double filter_bound;

JobNonceTracker::JobNonceTracker(job::Job j)
    : j(std::move(j))
    , remaining(size_t(std::numeric_limits<uint32_t>::max()) + 1)
    , offset(randuint32())
{
}

auto JobNonceTracker::get_job(size_t N) -> std::optional<JobNonceRange>
{
    N = std::min(N, remaining);
    if (N == 0)
        return {};
    remaining -= N;
    auto tmpOffset { offset };
    offset += uint32_t(N);
    return JobNonceRange {
        .j { j },
        .offset = tmpOffset,
        .N = uint32_t(N)
    };
};

namespace {
[[nodiscard]] auto build_program(cl::Context context)
{
    std::string code { kernel, strlen(kernel) - 1 };
    try {
        cl::Program program(context, std::string(code.begin(), code.end()));
        program.build("-cl-std=CL2.0 -DC_CONSTANT=" C_CONSTANT_STR);
        // program.build("-cl-std=CL2.0");
        return program;
    } catch (cl::BuildError& e) {
        auto logs { e.getBuildLog() };
        assert(logs.size() == 1);
        auto& log = logs[0].second;
        std::cerr << " Build error: " << log << std::endl
                  << std::flush;

        throw e;
    }
}
}

Sha256tGPUHasher::Sha256tGPUHasher(Sha256tOpenclHasher& parent, CL::Device& device)
    : deviceIndex(device.index())
    , deviceName { device.name() }
    , parent(parent)
    , context(device)
    , program(build_program(context))
    , functor(program, "h")
    , queue(context, device)
    , runner1(*this, N)
    , runner2(*this, N)
{
}
std::shared_ptr<CyclicQueue::Buffer> Sha256tGPUHasher::Allocator::allocateElements(size_t elements)
{
    assert(elements > 0);
    if (auto allocated = alloc(1 + 2 * elements); allocated) {
        assert((allocated->first.size() + allocated->second.size()) / 2 <= elements);
        return allocated;
    }
    return {};
}

using namespace std;

// void Sha256tGPUHasher::start_mining(Header h)
// {
//     cout << "Start mining" << endl;
//     std::lock_guard l(m);
//     active = true;
//     runner1.set_next_header(h);
//     runner2.set_next_header(h);
//     try_start();
// }
HashrateDelta Sha256tGPUHasher::stop_mining()
{
    std::lock_guard l(m);
    hashrateWatcher.reset();
    active = false;
    HashrateDelta hr { prevHashrate };
    prevHashrate = 0;
    return hr;
}

HashrateDelta Sha256tGPUHasher::set_zero()
{

    std::lock_guard l(m);
    HashrateDelta hd {
        .delta = -prevHashrate,
    };
    prevHashrate = 0;
    return hd;
};

double Sha256tGPUHasher::fraction() const
{
    return parent.fraction;
}
auto Sha256tGPUHasher::allocator() -> Allocator
{
    return parent.cyclicQueue->allocator();
}

void Sha256tGPUHasher::handle_finished_job(TripleSha::MinedValues mv) // is called by opencl callback thread
{

    // {
    //     union {
    //         uint32_t u32[20];
    //         uint8_t u8[80];
    //     } header;
    //     std::memcpy(header.u8, mv.header().data(), 76);
    //     // std::memset(header.u8, 0, 80);
    //     // cout<<"Current header: "<<serialize_hex(mv.header(),80)<<endl;
    //     // auto s = mv.sha256tValues->first;
    //     // cout<<"Read buffer: "<< serialize_hex((uint8_t*)s.data(),std::min(s.size(),10ul));
    //     auto sps { mv.result_spans() };
    //     for (auto& sp : sps.spans) {
    //         for (size_t i = 0; i < sp.size() && i < 1; ++i) {
    //             cout<<"Nonce: "<<sp[i].nonce()<<endl;
    //             header.u32[19] = hton32(sp[i].nonce());
    //             auto hs { hashSHA256(hashSHA256(hashSHA256(header.u8,80))) };
    //             uint32_t swapped = sp[i].hashStart();
    //             cout<<serialize_hex(hs)<<" "<<serialize_hex(swapped)<<"\n";
    //         }
    //     }
    //     return;
    // }

    // auto spans { mv.result_spans() };
    // uint32_t nsuccess(spans.size());
    // cout << "Success.size(): " << nsuccess << " N: " << N << endl;
    std::lock_guard l(m);
    auto duration { hashrateWatcher.register_hashes(N) };
    // if (duration) {
    //     cout << "duration: " << *duration / 1.0ms << " milliseconds" << endl;
    // }
    auto [currentHashrate, delta] = hashrate();
   // N = currentHashrate.val / 20; //
   N=gpubatchsize; // this is not right but somehow playing with N is giving unstable load on gpus
   // if (N < 1000)
   //     N = 1000;
    parent.handle_finished_job(std::move(mv), { delta });
    try_start();
}

std::pair<Hashrate, ssize_t> Sha256tGPUHasher::hashrate()
{
    ssize_t currentHashrate(hashrateWatcher.hashrate().val);
    ssize_t delta = ssize_t(currentHashrate) - prevHashrate;
    prevHashrate = currentHashrate;
    return { currentHashrate, delta };
}

void Sha256tGPUHasher::reset_start()
{
    {
        std::unique_lock l(m);
        active = true;
    }
    job = {};
    try_start();
}

std::optional<JobNonceRange> Sha256tGPUHasher::get_worker_range(size_t N)
{
    // try update job new
    if (!job.has_value()) {
        if (auto j = parent.generate_job(); j.has_value()) {
            job = JobNonceTracker { j.value() };
        } else
            return {};
    }

    if (auto j { job->get_job(N) }; j.has_value()) {
        if (job->exhausted())
            job.reset();
        return j;
    } else {
        job.reset();
        return {};
    }
}

void Sha256tGPUHasher::try_start()
{
    std::unique_lock l(m);
    if (active) {
        double f { fraction() };
        runner1.try_start(queue, f, N, functor);
       //runner2.try_start(queue, f, N, functor); //don't know why we need runner2, but without it seems works better, feels like extra load on pcie, maybe it's nvidia thing
    }
}

void Sha256tOpenclHasher::update_fraction_locked()
{
    using namespace std;
    using namespace std::chrono;
    if (!verushashrate)
        return;
    auto& vh(*verushashrate);
    

    if (autofiltering==true){ //autofiltering
        // if we tune filtering from current verus hashrate it leads to a loop of reducing filtering range, which eventualy leads to a lower verus hashrate than we could get
        // to avoid it we need to start calculating it from max verus and max sha256t hashrate that we have catched:

    	if(maxsha256th*1.05<totalHashrate){  //to keep max unfiltered sha256t hashrate that gpus can handle, 1.05 to smooth random spikes
        	maxsha256th=(totalHashrate+maxsha256th)/2; // to smooth 
        	//spdlog::info("new maxsha256th={}", maxsha256th); 
       		 gpufilterbound=log2( maxsha256th /(maxvh+ (maxsha256th/200)) );  // calculating gpufilterbound from max hashrate values
       		// spdlog::info("new gpufilterbound={}", gpufilterbound);   
   	 }
    
    	if (maxvh*1.08<verushashrate.value()){          // to keep max verus hashrate that cpu can handle, 1.08 to smooth random spikes,
    						        // as we start from gpufilterbound=1.0 cpu will have 100% load, so we definitely gonna get this value
                                                        // little problem: pcie bus and buffer load eats about 4-6% cpu hashrate so the real maxvh will be little lower
    		maxvh=(verushashrate.value()*0.97);     // reserve 3% (%?) for other work that cpu should handle (pcie load, buffer, pool connection etc)  
    		//spdlog::info("new maxvh={}", maxvh); 
    		if(totalHashrate==0){    //it happens sometimes (new block, lost connection etc) but cpu still have job in buffer, so we can catch maxvh that is way higher than cpu can sustain in normal conditions
    			gpufilterbound/=2; //this should restart tuning again 
    	       		maxvh/=2;          //this should restart tuning again 
    		}else{ 
    	        // calculating gpufilterbound from max hashrate values:	
    		gpufilterbound=log2( ((totalHashrate+maxsha256th)/2) /(maxvh+ ( ((totalHashrate+maxsha256th)/2) /200))); // using average sha256t hashrate from (totalHashrate+maxsha256th)/2 
    															 //TODO: to find more accurate upper limit sha256t hashrate from gpus
    		//spdlog::info("new gpufilterbound={}", gpufilterbound);   
    	  	 
    		}
    	}
    	
    	// filtered_sha256_stream. If we compare it to current verus hashrate, we can estimate are we sending to cpu too much or not enough:
    	filtered_sha256_stream=(totalHashrate/pow(2,gpufilterbound)-(totalHashrate/200)); 
    	
    	// mean of a filtered gpu to cpu function: (1/(0-"CPU"))*∫log2("GPU"/(y+("GPU"/200)))dy from "CPU" to 0 // see https://www.desmos.com/Calculator/lcjylqepyv for more details
    	// used only for observation
    	average_sha256t=((((totalHashrate+200*filtered_sha256_stream)*log(1+200*double(filtered_sha256_stream)/totalHashrate))/200)-(filtered_sha256_stream*(log(200)+1)))/(log(2)*(-filtered_sha256_stream)); 
    	
    	//finetune:
    	if(filtered_sha256_stream/verushashrate.value()>1.1){ //we are sending too much, not enough filtering 
    		gpufilterbound+=0.001; 			     //this should help to finetune
    	}
    	if(filtered_sha256_stream/verushashrate.value()<0.9){ //we are sending not enough, too much filtering 
    		gpufilterbound-=0.001;                       //this should help to finetune
    	}
    	
    }else{ // manual filtering
    	gpufilterbound=gpu_manual_filter_bound; // setting manual filtering from value given in arg
    	filtered_sha256_stream=(totalHashrate/pow(2,gpufilterbound)-(totalHashrate/200)); // // filtered_sha256_stream, with manual filtering we are using it only for observation
    	
    	if(totalHashrate!=0){ //with totalHashrate==0 in the next step there is nothing to calculate
    		// mean of a filtered gpu to cpu function: (1/(0-"CPU"))*∫log2("GPU"/(y+("GPU"/200)))dy from "CPU" to 0 // see https://www.desmos.com/Calculator/lcjylqepyv for more details
    		// used only for observation
    		average_sha256t=((((totalHashrate+200*filtered_sha256_stream)*log(1+200*double(filtered_sha256_stream)/totalHashrate))/200)-(filtered_sha256_stream*(log(200)+1)))/(log(2)*(-filtered_sha256_stream));
    	}
    }
    if (gpufilterbound<1){     // just in case
    	gpufilterbound=1; //not sure if we need this here but with values <1 filtering may be broken
    }   
    filter_bound=gpufilterbound; //setting calculated gpufilterbound
    
    if (totalHashrate == 0)
        fraction = 1.0;

    double tmp = double(verushashrate.value()) / double(totalHashrate);
    if (tmp > 1.0) {
        static optional<steady_clock::time_point> lastReported;
        auto now { steady_clock::now() };
        if (!lastReported || *lastReported + 10s < now) {
            spdlog::warn("CPU ({}/s verus v2.1) outruns GPUs ({}/s sha256t)", vh.format().to_string(), Hashrate(totalHashrate).format().to_string());
            lastReported = now;
        }
    }
    assert(tmp != 0.0);
    fraction = tmp;
}

void Sha256tOpenclHasher::allocation_possible()
{
    cleanSum = 0;
    for (auto& h : hashers) {
        h->try_start();
    }
}

std::pair<uint64_t, std::vector<std::tuple<uint32_t, std::string, uint64_t>>> Sha256tOpenclHasher::hashrates()
{
    std::lock_guard l(m);
    std::vector<std::tuple<uint32_t, std::string, uint64_t>> v;
    for (auto& h : hashers) {
        auto [hashrate, delta] = h->hashrate();
        totalHashrate += delta;
        v.push_back({ h->deviceIndex, h->deviceName, hashrate });
       // spdlog::info("totalHashrate={}, delta={}, h->deviceName={}, hashrate={} ", totalHashrate,delta, h->deviceName, hashrate);
       
        // TODO
        // After a long run sometimes total hashrate is calculated wrong (adds x2-3 more than it should be),
        // can't catch when it's happens, hashrates per devices is shown ok at the same time
        // it can ruin calculation of filtering bound
        // DONE: opencl_hasher.hpp:340
        // totalHashrate do not set to 0, if there is a drop in connection to a node and stop_mining happens. That's because delta is already set to 0.

        
                                                                                                               			  
    }
    return { totalHashrate, std::move(v) };
};

void Sha256tOpenclHasher::update_verushashrate(Hashrate hr)
{
    assert(hr.val != 0);
    std::lock_guard l(m);
    verushashrate = hr;
    update_fraction_locked();
}

void Sha256tOpenclHasher::set_work(job::GeneratorArg a)
{
    {
        std::lock_guard l(m);
        std::visit([&](auto& a) { jobGenerator.from_arg(std::move(a)); }, a);
    }
    for (auto& h : hashers)
        h->reset_start();
}
void Sha256tOpenclHasher::wakeup()
{
    for (auto& h : hashers) {
        h->try_start();
    }
};

void Sha256tOpenclHasher::handle_finished_job(TripleSha::MinedValues mined, HashrateDelta hd)
{
    {
        std::lock_guard l(m);
        totalHashrate += hd.delta;
    }
    on_mined(std::move(mined));
}

auto Sha256tOpenclHasher::generate_job() -> std::optional<Job>
{
    std::lock_guard l(m);
    return jobGenerator.generate();
}
