# Shared helper functions for init scripts
#
# Usage: source this file at the top of init scripts
#   . "$DIR/../lib/init-helpers.sh"

# Default to interactive mode
INTERACTIVE=${INTERACTIVE:-true}

# Prompt for confirmation. Returns 0 (true) if confirmed or non-interactive.
# Usage: confirm "Install X?" && do_something
confirm() {
  local prompt="$1"
  if [[ "$INTERACTIVE" != "true" ]]; then
    return 0
  fi
  read -p "$prompt [Y/n] " response
  [[ -z "$response" || "$response" =~ ^[Yy] ]]
}

# Print section header
section_header() {
  echo "---------------------------------------------"
  echo " $1 ..."
  echo "---------------------------------------------"
}

# Print section footer
section_footer() {
  echo "---------------------------------------------"
  echo " ... finished $1"
  echo "---------------------------------------------"
}
