# The training runs. `variant <name>` sets the variables of one run; each differs from base in the data only.
VARIANTS=${VARIANTS:-"base noRS log log_p2 noUNK noBEST noUNK_noBEST"}
variant() {
  unset NAME TAGS
  export DATASET=ALL_BEST EXCLUDE= Y_NORM=causal_log N_PREFIXES=0 TAG_DROPOUT=0.2
  case "$1" in
    base) ;;
    noRS) export EXCLUDE=RS ;;                       # no random search lines at all
    log) export Y_NORM=log ;;                        # rescaled log, full lines only
    log_p2) export Y_NORM=log N_PREFIXES=2 ;;        # rescaled log, two prefixes per line
    noUNK) export TAG_DROPOUT=0 ;;                   # every line keeps its tag
    noBEST) export DATASET=ALL ;;                    # no BEST lines; UNK is the only tag that names no optimizer
    noUNK_noBEST) export DATASET=ALL TAG_DROPOUT=0 ;;  # control: optimizer tags only
    *) echo "unknown variant $1" >&2; return 1 ;;
  esac
}
# Model seeds of a variant (SEEDS="..." overrides them for all).
seeds_of() {
  if [[ -n "${SEEDS:-}" ]]; then echo "$SEEDS"; return; fi
  case "$1" in
    base|noUNK|noBEST) echo 0 1 2 ;;
    *) echo 0 ;;
  esac
}
