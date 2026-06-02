#!/bin/bash

# ==============================================================================
# Benchmark Orchestration Script
# Runs the full vulnerability reproduction and patch lifecycle.
#
# Flow:
#   Phase 1 - Build vulnerable extension → trigger ASAN → capture trace
#   Phase 2 - Build patched extension   → run verification benchmarks
# ==============================================================================

# Exit on error EXCEPT where we explicitly handle it
set -e

# Ensure we are in project root
cd "$(dirname "$0")/.."

LOGFILE="benchmark_output.log"
echo "[*] Logging output to: $LOGFILE"
echo "" > "$LOGFILE"

echo "================================================================="
echo " PHASE 1: VULNERABILITY REPRODUCTION (ASAN TRIGGER)              "
echo "================================================================="

# Configure for vulnerable build
export PATCH_APPLIED=0
export ASAN_OPTIONS=detect_leaks=0:symbolize=1:halt_on_error=1

# Compile vulnerable version
./scripts/build.sh 2>&1 | tee -a "$LOGFILE"

# Run trigger script — we EXPECT this to fail (ASAN abort)
# Temporarily disable set -e so the script continues after the expected crash
echo ""
echo "[*] Executing Trigger Payload (Adversarial Tensor Shape)..."
echo "[*] Expecting ASAN to intercept a stack-buffer-overflow..."
echo ""

set +e
python test_trigger.py 2>&1 | tee -a "$LOGFILE"
TRIGGER_EXIT=$?
set -e

echo ""
if [ $TRIGGER_EXIT -ne 0 ]; then
    echo "[+] ✅ SUCCESS: AddressSanitizer caught the memory corruption (exit code: $TRIGGER_EXIT)"
    echo "REPRODUCIBILITY_STATUS=PASS" >> "$LOGFILE"
else
    echo "[-] ❌ FAIL: Vulnerability was not triggered or ASAN is not active."
    echo "REPRODUCIBILITY_STATUS=FAIL" >> "$LOGFILE"
fi


echo ""
echo "================================================================="
echo " PHASE 2: PATCH DEPLOYMENT & VERIFICATION BENCHMARKS             "
echo "================================================================="

# Configure for secure build
export PATCH_APPLIED=1
unset ASAN_OPTIONS

# Recompile secure version
./scripts/build.sh 2>&1 | tee -a "$LOGFILE"

# Run Verification Suite
echo ""
echo "[*] Executing Post-Fix Verification and Benchmark Suite..."
echo ""
python test_verification.py 2>&1 | tee -a "$LOGFILE"

echo ""
echo "================================================================="
echo " BENCHMARK LIFECYCLE COMPLETE                                    "
echo " Full output saved to: $LOGFILE                                  "
echo "================================================================="
