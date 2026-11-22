#!/usr/bin/env bash
# H42: build KenLM (lmplz, build_binary, query) in user space, no root and no system Boost.
#
#   lm_build_kenlm.sh <install-dir> <scratch-dir>
#
# <install-dir>  durable, e.g. /mnt/xbabe3-external/jsr/lm/tools/kenlm (exFAT is fine: only regular
#                files are written there: the two source archives, their sha256, and bin/).
# <scratch-dir>  POSIX filesystem for the build tree (the data drive is exFAT: no symlinks), e.g.
#                /dev/shm/lm-lane-build. It is deleted at the end; nothing durable lives there.
#
# Boost is built from the release archive (static program_options/thread/system only) and KenLM from
# a pinned commit, so the binaries need nothing but the system libstdc++/libz. Both archives are
# hash-checked: Boost against the sha256 pinned here, KenLM by its commit id (the archive's sha256
# is recorded next to it). Pin CPUs from outside (taskset); JOBS sets the compile width.
set -euo pipefail
OUT=${1:?install dir}
SCRATCH=${2:?scratch build dir}
JOBS=${JOBS:-14}
BOOST_VER=1.84.0
BOOST_TAR=boost_1_84_0.tar.gz
BOOST_SHA256=a5800f405508f5df8114558ca9855d2640a2de8f0445f051fa1c7c3383045724
KENLM_COMMIT=4cb443e60b7bf2c0ddf3c745378f76cb59e254e5
MAX_ORDER=10

mkdir -p "$OUT/src/boost.d" "$OUT/src/kenlm.d" "$OUT/bin" "$SCRATCH"
boost="$OUT/src/boost.d/$BOOST_TAR"
kenlm="$OUT/src/kenlm.d/kenlm-$KENLM_COMMIT.tar.gz"
[ -s "$boost" ] || curl -sS -L --fail --retry 5 -o "$boost" \
  "https://archives.boost.io/release/$BOOST_VER/source/$BOOST_TAR"
echo "$BOOST_SHA256  $boost" | sha256sum -c -
[ -s "$kenlm" ] || curl -sS -L --fail --retry 5 -o "$kenlm" \
  "https://github.com/kpu/kenlm/archive/$KENLM_COMMIT.tar.gz"
sha256sum "$kenlm" | tee "$OUT/src/kenlm.d/SHA256"

cd "$SCRATCH"
# Headers plus the libraries b2 has to compile (thread pulls atomic/chrono/date_time sources).
tar -xzf "$boost" --wildcards 'boost_1_84_0/boost' 'boost_1_84_0/tools/build' \
  'boost_1_84_0/tools/boost_install' 'boost_1_84_0/libs/config' 'boost_1_84_0/libs/headers' \
  'boost_1_84_0/libs/program_options' 'boost_1_84_0/libs/thread' 'boost_1_84_0/libs/system' \
  'boost_1_84_0/libs/atomic' 'boost_1_84_0/libs/chrono' 'boost_1_84_0/libs/date_time' \
  'boost_1_84_0/libs/predef' 'boost_1_84_0/libs/Jamfile.v2' 'boost_1_84_0/bootstrap.sh' \
  'boost_1_84_0/Jamroot' 'boost_1_84_0/boost-build.jam' 'boost_1_84_0/boostcpp.jam' \
  'boost_1_84_0/project-config.jam' 'boost_1_84_0/libs/numeric' 'boost_1_84_0/libs/detail' \
  2>/dev/null || tar -xzf "$boost"
cd boost_1_84_0
./bootstrap.sh --with-libraries=program_options,thread,system >/dev/null
./b2 -j"$JOBS" -d0 link=static variant=release threading=multi cxxflags=-fPIC stage
cd "$SCRATCH"
mkdir -p kenlm && tar -xzf "$kenlm" -C kenlm --strip-components=1
# The test framework is only needed for KenLM's unit tests; drop it from the required components.
sed -i 's/^\( *\)unit_test_framework$/\1/' kenlm/CMakeLists.txt
cmake -S kenlm -B kenlm/build -DCMAKE_BUILD_TYPE=Release -DKENLM_MAX_ORDER=$MAX_ORDER \
  -DBOOST_ROOT="$SCRATCH/boost_1_84_0" -DBoost_NO_SYSTEM_PATHS=ON -DBoost_USE_STATIC_LIBS=ON \
  -DBUILD_TESTING=OFF -DCOMPILE_TESTS=OFF -DCMAKE_DISABLE_FIND_PACKAGE_Eigen3=ON \
  -DBUILD_PYTHON=OFF >/dev/null
cmake --build kenlm/build -j"$JOBS" --target lmplz build_binary query
for b in lmplz build_binary query; do
  install -m 0755 "kenlm/build/bin/$b" "$OUT/bin/$b"
done
{
  echo "kenlm_commit=$KENLM_COMMIT max_order=$MAX_ORDER boost=$BOOST_VER"
  g++ --version | head -1
  (cd "$OUT/bin" && sha256sum lmplz build_binary query)
} > "$OUT/BUILD_INFO"
cat "$OUT/BUILD_INFO"
cd / && rm -rf "$SCRATCH"
