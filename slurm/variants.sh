# The training runs. `variant <name>` sets the variables of one run; each differs from base in the data only.
VARIANTS=${VARIANTS:-"base noRS log log_p2"}
variant() {
  unset NAME TAGS
  export DATASET=ALL_BEST EXCLUDE= Y_NORM=causal_log N_PREFIXES=0
  case "$1" in
    base) ;;
    noRS) export EXCLUDE=RS ;;                       # no random search lines at all
    log) export Y_NORM=log ;;                        # rescaled log, full lines only
    log_p2) export Y_NORM=log N_PREFIXES=2 ;;        # rescaled log, two prefixes per line
    *) echo "unknown variant $1" >&2; return 1 ;;
  esac
}
