#include "api_call.hpp"
#include "cmdline/cmdline.h"
#include "crypto/address.hpp"
#include "crypto/verushash/verushash.hpp"
#include "device_pool.hpp"
#include "general/hex.hpp"
#include "log/trace.hpp"
#include "logs.hpp"
#include "spdlog/spdlog.h"
#include "stratum/connection.hpp"
#include <cmath>
#include <iostream>
#include <variant>
using namespace std;

int start_miner(std::string gpus, size_t threads, ConnectionArg connectionData);

int test_gpu_miner2();

// not a right way to place it here, TODO: to make a cleaner code)
uint32_t gpubatchsize;   //for fixed gpubatchsize , somehow fixed value gives more stable load on gpus, 20000000 is default,
			// tested this value with different combinations of 1-3 gpus (rtx 2070-3070ti-3090) paired with ryzen 5950x and 5700x, works well enough // TODO: autofinder for optimal gpubatchsize
double gpu_manual_filter_bound; //for fixed manual filtering bound,value as pow2, without "-" // TODO: remove gpu_manual_filter_bound, instead use filter_bound and bool for auto filtering
double filter_bound=1; // for filter bound, as pow2, without "-" 
bool autofiltering; // if no filtering value is given in args, then autofiltering=true

int process(gengetopt_args_info& ai)
{
    try {
        std::string host { ai.host_arg };
        uint16_t port(ai.port_arg);
        if (ai.threads_arg < 0)
            throw std::runtime_error("Illegal value " + to_string(ai.threads_arg) + " for option --threads.");
        size_t threads { ai.threads_arg == 0 ? std::thread::hardware_concurrency() : ai.threads_arg };
        // Address address(ai.address_arg);
        
        if(ai.gpubatchsize_given){ //checking if value for manual Gpu batchsize is given
        		if(ai.gpubatchsize_arg<1){ 
        			spdlog::error("Gpu batchsize can't be <1");
            			return -1;
        				
        		}
        		if(ai.gpubatchsize_arg<10000000){  //if < 20ms it won't trigger update_fraction_locked() and nothing will work, but maybe there are some super-slow opencl devices) 
        			spdlog::warn("Gpu batchsize is too low, mining may not work as intended");  
        		}
        }
        gpubatchsize=ai.gpubatchsize_arg; // setting fixed batchsize for gpus
        cout << "Fixed batchsize for gpus:" << gpubatchsize << endl;
        
        if(ai.gpufilter_given){  //checking if value for manual filtering is given
        	if (1<=ai.gpufilter_arg && ai.gpufilter_arg<=7.64){  //checking for correct values within 1.0 to 7.64 range
                autofiltering=false; 
                gpu_manual_filter_bound=ai.gpufilter_arg;
        	cout << "Manual sha256t filtering enabled with filtering bound:" << ai.gpufilter_arg << endl;
        	}else{
        		spdlog::error("Sha256t filtering bound should be in 1.00 to 7.64 range");
        		return -1;
        	}
        }else{
                autofiltering=true;
        	cout << "Auto-filtering for sha256t enabled" << endl;
        }

                
        std::string gpus;
        if (ai.gpus_given) {
            gpus.assign(ai.gpus_arg);
        }

        if (ai.queuesize_arg < 0) {
            spdlog::error("Queue size cannot be negative");
        }
        if (ai.address_given && strlen(ai.address_arg) > 0) {
            if (ai.user_given && strlen(ai.user_arg) > 0)
                spdlog::warn("Stratum parameter '-u' is ignored because direct-to-node mining is enabled via '-a'");
            start_miner(gpus, threads,
                NodeConnectionData {
                    .host { host },
                    .port = port,
                    .queuesizeGB = static_cast<size_t>(ai.queuesize_arg),
                    .address { ai.address_arg } });
        } else if (ai.user_given) { // stratum
            start_miner(gpus, threads,
                stratum::ConnectionData {
                    .host { ai.host_arg },
                    .port { std::to_string(ai.port_arg) },
                    .queuesizeGB = static_cast<size_t>(ai.queuesize_arg),
                    .user { ai.user_arg },
                    .pass { ai.password_arg } });
        } else {
            spdlog::error("Either -a or -u parameter is required");
            return -1;
        }
    } catch (std::runtime_error& e) {
        spdlog::error("{}", e.what());
        return -1;
    } catch (Error& e) {
        spdlog::error("{}", e.strerror());
        return -1;
    }
    return 0;
}

int main(int argc, char** argv)
{
    initialize_mining_log();
    srand(time(0));
    cout << "¯\\_(ツ)_/¯" << endl;
    cout << "Janushash Miner Plus (99.9% based on Janushash Miner by CoinFuMasterShifu) ⚒ ⛏" << endl;
    gengetopt_args_info ai;
    if (cmdline_parser(argc, argv, &ai) != 0) {
        return -1;
    }
    int i = process(ai);
    cmdline_parser_free(&ai);
    return i;
}
