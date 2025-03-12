#!/usr/bin/env bash

mkdir -p ~/Developer
PROFILE_DIR="~/Developer/alexdlaird"

if [ -d "$PROFILE_DIR" ]; then
  git -C $PROFILE_DIR fetch
  git -C $PROFILE_DIR pull
else
  git clone git@github.com:alexdlaird/alexdlaird.git $PROFILE_DIR
fi

export PATH=$PATH:$PROFILE_DIR/tools/bin

. $PROFILE_DIR/tools/init/dev-init-system

. $PROFILE_DIR/tools/bin/dev-repos-init
