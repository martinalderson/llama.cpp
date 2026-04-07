#!/bin/bash
set -e
cmake --build build-vk --target llama-moe-profile -j $(nproc) 2>&1 | tail -3
./build-vk/bin/llama-moe-profile \
  -m /home/martin/.lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf \
  -ngl 12 \
  -p "Write a Python implementation of a B-tree data structure with insert, search, and delete operations. Include type hints and docstrings." \
  -n 1024 -c 4096 \
  --temp 0.7 --repeat-penalty 1.1 \
  2>/dev/null && echo "DONE - reload /tmp/moe-profile.html"
