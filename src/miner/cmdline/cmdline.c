/*
  File autogenerated by gengetopt version 2.23
  generated with the following command:
  gengetopt -i cmdoptions.ggo 

  The developers of gengetopt consider the fixed text that goes in all
  gengetopt output files to be in the public domain:
  we make no copyright claims on it.
*/

/* If we use autoconf.  */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef FIX_UNUSED
#define FIX_UNUSED(X) (void) (X) /* avoid warnings for unused params */
#endif

#include <getopt.h>

#include "cmdline.h"

const char *gengetopt_args_info_purpose = "GPU/CPU miner of the Warthog Network supporting algorithm change to Janushash.\n\n\nBy CoinFuMasterShifu, Pumbaa, Timon & Rafiki";

const char *gengetopt_args_info_usage = "Usage: wart-miner-janushash [OPTION]...";

const char *gengetopt_args_info_versiontext = "";

const char *gengetopt_args_info_description = "";

const char *gengetopt_args_info_help[] = {
  "      --help                   Print help and exit",
  "  -V, --version                Print version and exit",
  "      --gpus=STRING            Specify GPUs as comma separated list like\n                                 \"0,2,3\". By default all GPUs are used.",
  "  -t, --threads=INT            Number of CPU worker threads, use 0 for hardware\n                                 concurrency.   (default=`0')",
  "  -h, --host=STRING            Host (RPC-Node)  (default=`localhost')",
  "  -p, --port=INT               Port (RPC-Node)  (default=`3000')",
  "  -g, --gpubatchsize=INT       Gpu batchsize  (default=`20000000')",
  "  -f, --gpufilter=DOUBLE       Manual sha256t filtering bound, filtering will\n                                 be from 2^(-gpufilter) to 2^(-7.64), specify\n                                 without \"-\" like \"5.5\"   (default=`1.0')",
  "  -a, --address=WALLETADDRESS  Specify address that is mined to (for mining\n                                 directly to node)  (default=`')",
  "  -q, --queuesize=INT          Queuesize in GB  (default=`4')",
  "  -u, --user=STRING            Enable stratum protocol and specify username\n                                 (default=`')",
  "      --password=STRING        Password (for Stratum)  (default=`')",
    0
};

typedef enum {ARG_NO
  , ARG_STRING
  , ARG_INT
  , ARG_DOUBLE
} cmdline_parser_arg_type;

static
void clear_given (struct gengetopt_args_info *args_info);
static
void clear_args (struct gengetopt_args_info *args_info);

static int
cmdline_parser_internal (int argc, char **argv, struct gengetopt_args_info *args_info,
                        struct cmdline_parser_params *params, const char *additional_error);


static char *
gengetopt_strdup (const char *s);

static
void clear_given (struct gengetopt_args_info *args_info)
{
  args_info->help_given = 0 ;
  args_info->version_given = 0 ;
  args_info->gpus_given = 0 ;
  args_info->threads_given = 0 ;
  args_info->host_given = 0 ;
  args_info->port_given = 0 ;
  args_info->gpubatchsize_given = 0 ;
  args_info->gpufilter_given = 0 ;
  args_info->address_given = 0 ;
  args_info->queuesize_given = 0 ;
  args_info->user_given = 0 ;
  args_info->password_given = 0 ;
}

static
void clear_args (struct gengetopt_args_info *args_info)
{
  FIX_UNUSED (args_info);
  args_info->gpus_arg = NULL;
  args_info->gpus_orig = NULL;
  args_info->threads_arg = 0;
  args_info->threads_orig = NULL;
  args_info->host_arg = gengetopt_strdup ("localhost");
  args_info->host_orig = NULL;
  args_info->port_arg = 3000;
  args_info->port_orig = NULL;
  args_info->gpubatchsize_arg = 20000000;
  args_info->gpubatchsize_orig = NULL;
  args_info->gpufilter_arg = 1.0;
  args_info->gpufilter_orig = NULL;
  args_info->address_arg = gengetopt_strdup ("");
  args_info->address_orig = NULL;
  args_info->queuesize_arg = 4;
  args_info->queuesize_orig = NULL;
  args_info->user_arg = gengetopt_strdup ("");
  args_info->user_orig = NULL;
  args_info->password_arg = gengetopt_strdup ("");
  args_info->password_orig = NULL;
  
}

static
void init_args_info(struct gengetopt_args_info *args_info)
{


  args_info->help_help = gengetopt_args_info_help[0] ;
  args_info->version_help = gengetopt_args_info_help[1] ;
  args_info->gpus_help = gengetopt_args_info_help[2] ;
  args_info->threads_help = gengetopt_args_info_help[3] ;
  args_info->host_help = gengetopt_args_info_help[4] ;
  args_info->port_help = gengetopt_args_info_help[5] ;
  args_info->gpubatchsize_help = gengetopt_args_info_help[6] ;
  args_info->gpufilter_help = gengetopt_args_info_help[7] ;
  args_info->address_help = gengetopt_args_info_help[8] ;
  args_info->queuesize_help = gengetopt_args_info_help[9] ;
  args_info->user_help = gengetopt_args_info_help[10] ;
  args_info->password_help = gengetopt_args_info_help[11] ;
  
}

void
cmdline_parser_print_version (void)
{
  printf ("%s %s\n",
     (strlen(CMDLINE_PARSER_PACKAGE_NAME) ? CMDLINE_PARSER_PACKAGE_NAME : CMDLINE_PARSER_PACKAGE),
     CMDLINE_PARSER_VERSION);

  if (strlen(gengetopt_args_info_versiontext) > 0)
    printf("\n%s\n", gengetopt_args_info_versiontext);
}

static void print_help_common(void)
{
	size_t len_purpose = strlen(gengetopt_args_info_purpose);
	size_t len_usage = strlen(gengetopt_args_info_usage);

	if (len_usage > 0) {
		printf("%s\n", gengetopt_args_info_usage);
	}
	if (len_purpose > 0) {
		printf("%s\n", gengetopt_args_info_purpose);
	}

	if (len_usage || len_purpose) {
		printf("\n");
	}

	if (strlen(gengetopt_args_info_description) > 0) {
		printf("%s\n\n", gengetopt_args_info_description);
	}
}

void
cmdline_parser_print_help (void)
{
  int i = 0;
  print_help_common();
  while (gengetopt_args_info_help[i])
    printf("%s\n", gengetopt_args_info_help[i++]);
}

void
cmdline_parser_init (struct gengetopt_args_info *args_info)
{
  clear_given (args_info);
  clear_args (args_info);
  init_args_info (args_info);
}

void
cmdline_parser_params_init(struct cmdline_parser_params *params)
{
  if (params)
    { 
      params->override = 0;
      params->initialize = 1;
      params->check_required = 1;
      params->check_ambiguity = 0;
      params->print_errors = 1;
    }
}

struct cmdline_parser_params *
cmdline_parser_params_create(void)
{
  struct cmdline_parser_params *params = 
    (struct cmdline_parser_params *)malloc(sizeof(struct cmdline_parser_params));
  cmdline_parser_params_init(params);  
  return params;
}

static void
free_string_field (char **s)
{
  if (*s)
    {
      free (*s);
      *s = 0;
    }
}


static void
cmdline_parser_release (struct gengetopt_args_info *args_info)
{

  free_string_field (&(args_info->gpus_arg));
  free_string_field (&(args_info->gpus_orig));
  free_string_field (&(args_info->threads_orig));
  free_string_field (&(args_info->host_arg));
  free_string_field (&(args_info->host_orig));
  free_string_field (&(args_info->port_orig));
  free_string_field (&(args_info->gpubatchsize_orig));
  free_string_field (&(args_info->gpufilter_orig));
  free_string_field (&(args_info->address_arg));
  free_string_field (&(args_info->address_orig));
  free_string_field (&(args_info->queuesize_orig));
  free_string_field (&(args_info->user_arg));
  free_string_field (&(args_info->user_orig));
  free_string_field (&(args_info->password_arg));
  free_string_field (&(args_info->password_orig));
  
  

  clear_given (args_info);
}


static void
write_into_file(FILE *outfile, const char *opt, const char *arg, const char *values[])
{
  FIX_UNUSED (values);
  if (arg) {
    fprintf(outfile, "%s=\"%s\"\n", opt, arg);
  } else {
    fprintf(outfile, "%s\n", opt);
  }
}


int
cmdline_parser_dump(FILE *outfile, struct gengetopt_args_info *args_info)
{
  int i = 0;

  if (!outfile)
    {
      fprintf (stderr, "%s: cannot dump options to stream\n", CMDLINE_PARSER_PACKAGE);
      return EXIT_FAILURE;
    }

  if (args_info->help_given)
    write_into_file(outfile, "help", 0, 0 );
  if (args_info->version_given)
    write_into_file(outfile, "version", 0, 0 );
  if (args_info->gpus_given)
    write_into_file(outfile, "gpus", args_info->gpus_orig, 0);
  if (args_info->threads_given)
    write_into_file(outfile, "threads", args_info->threads_orig, 0);
  if (args_info->host_given)
    write_into_file(outfile, "host", args_info->host_orig, 0);
  if (args_info->port_given)
    write_into_file(outfile, "port", args_info->port_orig, 0);
  if (args_info->gpubatchsize_given)
    write_into_file(outfile, "gpubatchsize", args_info->gpubatchsize_orig, 0);
  if (args_info->gpufilter_given)
    write_into_file(outfile, "gpufilter", args_info->gpufilter_orig, 0);
  if (args_info->address_given)
    write_into_file(outfile, "address", args_info->address_orig, 0);
  if (args_info->queuesize_given)
    write_into_file(outfile, "queuesize", args_info->queuesize_orig, 0);
  if (args_info->user_given)
    write_into_file(outfile, "user", args_info->user_orig, 0);
  if (args_info->password_given)
    write_into_file(outfile, "password", args_info->password_orig, 0);
  

  i = EXIT_SUCCESS;
  return i;
}

int
cmdline_parser_file_save(const char *filename, struct gengetopt_args_info *args_info)
{
  FILE *outfile;
  int i = 0;

  outfile = fopen(filename, "w");

  if (!outfile)
    {
      fprintf (stderr, "%s: cannot open file for writing: %s\n", CMDLINE_PARSER_PACKAGE, filename);
      return EXIT_FAILURE;
    }

  i = cmdline_parser_dump(outfile, args_info);
  fclose (outfile);

  return i;
}

void
cmdline_parser_free (struct gengetopt_args_info *args_info)
{
  cmdline_parser_release (args_info);
}

/** @brief replacement of strdup, which is not standard */
char *
gengetopt_strdup (const char *s)
{
  char *result = 0;
  if (!s)
    return result;

  result = (char*)malloc(strlen(s) + 1);
  if (result == (char*)0)
    return (char*)0;
  strcpy(result, s);
  return result;
}

int
cmdline_parser (int argc, char **argv, struct gengetopt_args_info *args_info)
{
  return cmdline_parser2 (argc, argv, args_info, 0, 1, 1);
}

int
cmdline_parser_ext (int argc, char **argv, struct gengetopt_args_info *args_info,
                   struct cmdline_parser_params *params)
{
  int result;
  result = cmdline_parser_internal (argc, argv, args_info, params, 0);

  if (result == EXIT_FAILURE)
    {
      cmdline_parser_free (args_info);
      exit (EXIT_FAILURE);
    }
  
  return result;
}

int
cmdline_parser2 (int argc, char **argv, struct gengetopt_args_info *args_info, int override, int initialize, int check_required)
{
  int result;
  struct cmdline_parser_params params;
  
  params.override = override;
  params.initialize = initialize;
  params.check_required = check_required;
  params.check_ambiguity = 0;
  params.print_errors = 1;

  result = cmdline_parser_internal (argc, argv, args_info, &params, 0);

  if (result == EXIT_FAILURE)
    {
      cmdline_parser_free (args_info);
      exit (EXIT_FAILURE);
    }
  
  return result;
}

int
cmdline_parser_required (struct gengetopt_args_info *args_info, const char *prog_name)
{
  FIX_UNUSED (args_info);
  FIX_UNUSED (prog_name);
  return EXIT_SUCCESS;
}


static char *package_name = 0;

/**
 * @brief updates an option
 * @param field the generic pointer to the field to update
 * @param orig_field the pointer to the orig field
 * @param field_given the pointer to the number of occurrence of this option
 * @param prev_given the pointer to the number of occurrence already seen
 * @param value the argument for this option (if null no arg was specified)
 * @param possible_values the possible values for this option (if specified)
 * @param default_value the default value (in case the option only accepts fixed values)
 * @param arg_type the type of this option
 * @param check_ambiguity @see cmdline_parser_params.check_ambiguity
 * @param override @see cmdline_parser_params.override
 * @param no_free whether to free a possible previous value
 * @param multiple_option whether this is a multiple option
 * @param long_opt the corresponding long option
 * @param short_opt the corresponding short option (or '-' if none)
 * @param additional_error possible further error specification
 */
static
int update_arg(void *field, char **orig_field,
               unsigned int *field_given, unsigned int *prev_given, 
               char *value, const char *possible_values[],
               const char *default_value,
               cmdline_parser_arg_type arg_type,
               int check_ambiguity, int override,
               int no_free, int multiple_option,
               const char *long_opt, char short_opt,
               const char *additional_error)
{
  char *stop_char = 0;
  const char *val = value;
  int found;
  char **string_field;
  FIX_UNUSED (field);

  stop_char = 0;
  found = 0;

  if (!multiple_option && prev_given && (*prev_given || (check_ambiguity && *field_given)))
    {
      if (short_opt != '-')
        fprintf (stderr, "%s: `--%s' (`-%c') option given more than once%s\n", 
               package_name, long_opt, short_opt,
               (additional_error ? additional_error : ""));
      else
        fprintf (stderr, "%s: `--%s' option given more than once%s\n", 
               package_name, long_opt,
               (additional_error ? additional_error : ""));
      return 1; /* failure */
    }

  FIX_UNUSED (default_value);
    
  if (field_given && *field_given && ! override)
    return 0;
  if (prev_given)
    (*prev_given)++;
  if (field_given)
    (*field_given)++;
  if (possible_values)
    val = possible_values[found];

  switch(arg_type) {
  case ARG_INT:
    if (val) *((int *)field) = strtol (val, &stop_char, 0);
    break;
  case ARG_DOUBLE:
    if (val) *((double *)field) = strtod (val, &stop_char);
    break;
  case ARG_STRING:
    if (val) {
      string_field = (char **)field;
      if (!no_free && *string_field)
        free (*string_field); /* free previous string */
      *string_field = gengetopt_strdup (val);
    }
    break;
  default:
    break;
  };

  /* check numeric conversion */
  switch(arg_type) {
  case ARG_INT:
  case ARG_DOUBLE:
    if (val && !(stop_char && *stop_char == '\0')) {
      fprintf(stderr, "%s: invalid numeric value: %s\n", package_name, val);
      return 1; /* failure */
    }
    break;
  default:
    ;
  };

  /* store the original value */
  switch(arg_type) {
  case ARG_NO:
    break;
  default:
    if (value && orig_field) {
      if (no_free) {
        *orig_field = value;
      } else {
        if (*orig_field)
          free (*orig_field); /* free previous string */
        *orig_field = gengetopt_strdup (value);
      }
    }
  };

  return 0; /* OK */
}


int
cmdline_parser_internal (
  int argc, char **argv, struct gengetopt_args_info *args_info,
                        struct cmdline_parser_params *params, const char *additional_error)
{
  int c;	/* Character of the parsed option.  */

  int error_occurred = 0;
  struct gengetopt_args_info local_args_info;
  
  int override;
  int initialize;
  int check_required;
  int check_ambiguity;
  
  package_name = argv[0];
  
  /* TODO: Why is this here? It is not used anywhere. */
  override = params->override;
  FIX_UNUSED(override);

  initialize = params->initialize;
  check_required = params->check_required;

  /* TODO: Why is this here? It is not used anywhere. */
  check_ambiguity = params->check_ambiguity;
  FIX_UNUSED(check_ambiguity);

  if (initialize)
    cmdline_parser_init (args_info);

  cmdline_parser_init (&local_args_info);

  optarg = 0;
  optind = 0;
  opterr = params->print_errors;
  optopt = '?';

  while (1)
    {
      int option_index = 0;

      static struct option long_options[] = {
        { "help",	0, NULL, 0 },
        { "version",	0, NULL, 'V' },
        { "gpus",	1, NULL, 0 },
        { "threads",	1, NULL, 't' },
        { "host",	1, NULL, 'h' },
        { "port",	1, NULL, 'p' },
        { "gpubatchsize",	1, NULL, 'g' },
        { "gpufilter",	1, NULL, 'f' },
        { "address",	1, NULL, 'a' },
        { "queuesize",	1, NULL, 'q' },
        { "user",	1, NULL, 'u' },
        { "password",	1, NULL, 0 },
        { 0,  0, 0, 0 }
      };

      c = getopt_long (argc, argv, "Vt:h:p:g:f:a:q:u:", long_options, &option_index);

      if (c == -1) break;	/* Exit from `while (1)' loop.  */

      switch (c)
        {
        case 'V':	/* Print version and exit.  */
          cmdline_parser_print_version ();
          cmdline_parser_free (&local_args_info);
          exit (EXIT_SUCCESS);

        case 't':	/* Number of CPU worker threads, use 0 for hardware concurrency. .  */
        
        
          if (update_arg( (void *)&(args_info->threads_arg), 
               &(args_info->threads_orig), &(args_info->threads_given),
              &(local_args_info.threads_given), optarg, 0, "0", ARG_INT,
              check_ambiguity, override, 0, 0,
              "threads", 't',
              additional_error))
            goto failure;
        
          break;
        case 'h':	/* Host (RPC-Node).  */
        
        
          if (update_arg( (void *)&(args_info->host_arg), 
               &(args_info->host_orig), &(args_info->host_given),
              &(local_args_info.host_given), optarg, 0, "localhost", ARG_STRING,
              check_ambiguity, override, 0, 0,
              "host", 'h',
              additional_error))
            goto failure;
        
          break;
        case 'p':	/* Port (RPC-Node).  */
        
        
          if (update_arg( (void *)&(args_info->port_arg), 
               &(args_info->port_orig), &(args_info->port_given),
              &(local_args_info.port_given), optarg, 0, "3000", ARG_INT,
              check_ambiguity, override, 0, 0,
              "port", 'p',
              additional_error))
            goto failure;
        
          break;
        case 'g':	/* Gpu batchsize.  */
        
        
          if (update_arg( (void *)&(args_info->gpubatchsize_arg), 
               &(args_info->gpubatchsize_orig), &(args_info->gpubatchsize_given),
              &(local_args_info.gpubatchsize_given), optarg, 0, "20000000", ARG_INT,
              check_ambiguity, override, 0, 0,
              "gpubatchsize", 'g',
              additional_error))
            goto failure;
        
          break;
        case 'f':	/* Manual sha256t filtering bound, filtering will be from 2^(-gpufilter) to 2^(-7.64), specify without \"-\" like \"5.5\" .  */
        
        
          if (update_arg( (void *)&(args_info->gpufilter_arg), 
               &(args_info->gpufilter_orig), &(args_info->gpufilter_given),
              &(local_args_info.gpufilter_given), optarg, 0, "1.0", ARG_DOUBLE,
              check_ambiguity, override, 0, 0,
              "gpufilter", 'f',
              additional_error))
            goto failure;
        
          break;
        case 'a':	/* Specify address that is mined to (for mining directly to node).  */
        
        
          if (update_arg( (void *)&(args_info->address_arg), 
               &(args_info->address_orig), &(args_info->address_given),
              &(local_args_info.address_given), optarg, 0, "", ARG_STRING,
              check_ambiguity, override, 0, 0,
              "address", 'a',
              additional_error))
            goto failure;
        
          break;
        case 'q':	/* Queuesize in GB.  */
        
        
          if (update_arg( (void *)&(args_info->queuesize_arg), 
               &(args_info->queuesize_orig), &(args_info->queuesize_given),
              &(local_args_info.queuesize_given), optarg, 0, "4", ARG_INT,
              check_ambiguity, override, 0, 0,
              "queuesize", 'q',
              additional_error))
            goto failure;
        
          break;
        case 'u':	/* Enable stratum protocol and specify username.  */
        
        
          if (update_arg( (void *)&(args_info->user_arg), 
               &(args_info->user_orig), &(args_info->user_given),
              &(local_args_info.user_given), optarg, 0, "", ARG_STRING,
              check_ambiguity, override, 0, 0,
              "user", 'u',
              additional_error))
            goto failure;
        
          break;

        case 0:	/* Long option with no short option */
          if (strcmp (long_options[option_index].name, "help") == 0) {
            cmdline_parser_print_help ();
            cmdline_parser_free (&local_args_info);
            exit (EXIT_SUCCESS);
          }

          /* Specify GPUs as comma separated list like \"0,2,3\". By default all GPUs are used..  */
          if (strcmp (long_options[option_index].name, "gpus") == 0)
          {
          
          
            if (update_arg( (void *)&(args_info->gpus_arg), 
                 &(args_info->gpus_orig), &(args_info->gpus_given),
                &(local_args_info.gpus_given), optarg, 0, 0, ARG_STRING,
                check_ambiguity, override, 0, 0,
                "gpus", '-',
                additional_error))
              goto failure;
          
          }
          /* Password (for Stratum).  */
          else if (strcmp (long_options[option_index].name, "password") == 0)
          {
          
          
            if (update_arg( (void *)&(args_info->password_arg), 
                 &(args_info->password_orig), &(args_info->password_given),
                &(local_args_info.password_given), optarg, 0, "", ARG_STRING,
                check_ambiguity, override, 0, 0,
                "password", '-',
                additional_error))
              goto failure;
          
          }
          
          break;
        case '?':	/* Invalid option.  */
          /* `getopt_long' already printed an error message.  */
          goto failure;

        default:	/* bug: option not considered.  */
          fprintf (stderr, "%s: option unknown: %c%s\n", CMDLINE_PARSER_PACKAGE, c, (additional_error ? additional_error : ""));
          abort ();
        } /* switch */
    } /* while */



	FIX_UNUSED(check_required);

  cmdline_parser_release (&local_args_info);

  if ( error_occurred )
    return (EXIT_FAILURE);

  return 0;

failure:
  
  cmdline_parser_release (&local_args_info);
  return (EXIT_FAILURE);
}
/* vim: set ft=c noet ts=8 sts=8 sw=8 tw=80 nojs spell : */
