#!/bin/bash

RUSTLIB=librustlib.so
PYLIB=rustlib.so
SCRIPT_DIR=`dirname "$0"`
LIB_DIR=$SCRIPT_DIR/target/release
INSTALL_DIR=$SCRIPT_DIR/..

cargo +nightly build --release || exit $?
cp "$LIB_DIR/$RUSTLIB" "$INSTALL_DIR/$PYLIB" || exit $?
echo "Installed $INSTALL_DIR/$PYLIB"
