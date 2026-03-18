export NVM_DIR="$HOME/.nvm"

# Handle brew-installed nvm (macOS)
if [[ -s "$(brew --prefix 2>/dev/null)/opt/nvm/nvm.sh" ]]; then
  . "$(brew --prefix)/opt/nvm/nvm.sh"
  [[ -r "$(brew --prefix)/opt/nvm/etc/bash_completion.d/nvm" ]] && . "$(brew --prefix)/opt/nvm/etc/bash_completion.d/nvm"
# Handle manual nvm install
elif [[ -s "$NVM_DIR/nvm.sh" ]]; then
  . "$NVM_DIR/nvm.sh"
  [[ -r "$NVM_DIR/bash_completion" ]] && . "$NVM_DIR/bash_completion"
fi
