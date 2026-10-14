#!/bin/bash
D=$HOME/claude-scratch/bant-darbogaz
rm -f $D/spec_son_bitti.txt
$D/sb_qkv_kos.sh
$D/spec_son_kos.sh
