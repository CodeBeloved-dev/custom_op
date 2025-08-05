#!/bin/bash

dirs=("build" "b2" "b3")

for dir in "${dirs[@]}"; do
    [ -d "$dir" ] && rm -rf "$dir"
done
