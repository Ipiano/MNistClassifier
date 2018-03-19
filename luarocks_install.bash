#!/bin/bash
ret="$(luarocks list 2>&1 | grep $1)"

if [ -z "$ret" ]
then
    echo "Installing Lua dependency $1"
    luarocks install $1
else
    echo "Lua dependency $1 found"
fi
