# Datatype Recovery Experiments

This repository holds the wildebeest experiments I am running for datatype recovery.
There are several components that live in several different places (modified Ghidra,
wildebeest, phd repo, this repo...) and they may or may not ever get unified into
a single repository. For now, this serves as the top-level experiment and eventually
I may pull other pieces into this one depending on how things go.

## Install Pytorch
To install PyTorch 2.1.0/CUDA 11.8, use this command:

`pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118`

For different versions of CUDA or non-GPU versions, see alternate options here:
(from https://pytorch.org/get-started/previous-versions/#linux-and-windows-21)

## Setup
To run the `basic_dataset` experiment, Ghidra needs to already be installed and
configured. For now, the steps to set that up are:

`git clone git@github.com:lasserre/ghidra.git`

Install Gradle 7.3+ from: https://gradle.org/releases/

- Download the zip file, unzip, add the bin folder to `$PATH`

Install JDK 17 64-bit: https://adoptium.net/temurin/releases

- I already had this installed, but make sure it's on `$PATH` (it looks like I
just have the JDK folder itself on my path, *not* the `bin/` subfolder)

Then build Ghidra:

`gradle -I gradle/support/fetchDependencies.gradle init`

`gradle buildGhidra`

Install Ghidra (into e.g. `~/software/ghidra_10.3_DEV`) using:

`unzip build/dist/<FILENAME> -d ~/software`

## Sever Configuration
1. Configure the `server/server.conf` file in the new installation




~~~
# tmux new-window "<ghidra_install>/server/ghidraSvr console"
# ---------------------------------------------
# NOTE ghidra server should be configured with desired cmd-line options in
# its server.conf:
# ----------
# wrapper.java.maxmemory = 16 + (32 * FileCount/10000) + (2 * ClientCount)
# wrapper.java.maxmemory=XX   // in MB
# ghidra.repositories.dir=/home/cls0027/ghidra_server_projects
# <parameters>: -anonymous <ghidra.repositories.dir> OR
#               -a0 -e0 -u <ghidra.repositories.dir>
# ---------------------------------------------
~~~