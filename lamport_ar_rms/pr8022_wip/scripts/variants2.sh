#!/bin/bash
# usage: variants2.sh OUTDIR "M list" "N list" "name:CLUSTER:PACKS:SPLIT:SENDPACKS"...
out=$1; ms=$2; ns=$3; shift 3
cd /app/aiter
for v in "$@"; do
  IFS=: read name cl pk sp sd <<< "$v"
  AITER_LAMPORT_RMS_CLUSTER=$cl AITER_LAMPORT_RMS_PACKS=$pk AITER_LAMPORT_RMS_SPLIT=$sp AITER_LAMPORT_RMS_SENDPACKS=$sd PYTHONPATH=/tmp/agfull/noflydsl:/app/aiter \
    timeout 1800 python /root/.claude/projects/-app-aiter/notes/pr8022/bench.py -m $ms -n $ns -q 0 1 -c lamport --rounds 2 \
    > $out/$name.log 2>&1
  echo "$name rc=$?" >> $out/status.txt
done
