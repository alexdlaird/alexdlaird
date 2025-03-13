#!/usr/bin/env bash

mkdir -p $HOME/Developer
PROFILE_DIR="$HOME/Developer/alexdlaird"

if [ -d "$PROFILE_DIR" ]; then
  git -C $PROFILE_DIR fetch
  git -C $PROFILE_DIR pull
else
  git clone git@github.com:alexdlaird/alexdlaird.git $PROFILE_DIR
fi

. $PROFILE_DIR/tools/init/dev-init-system

. $PROFILE_DIR/tools/lib/rc-pyenv.sh

python $PROFILE_DIR/tools/bin/dev-repos-init

exec zsh
