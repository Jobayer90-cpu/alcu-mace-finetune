#!/bin/bash
# PSLibrary 1.0.0 PAW pseudopotentials, PBE. GPL - freely redistributable.
mkdir -p "$(dirname "$0")/../pseudos" && cd "$(dirname "$0")/../pseudos"
base=https://pseudopotentials.quantum-espresso.org/upf_files
wget -nc $base/Al.pbe-n-kjpaw_psl.1.0.0.UPF
wget -nc $base/Cu.pbe-dn-kjpaw_psl.1.0.0.UPF
sha256sum *.UPF
