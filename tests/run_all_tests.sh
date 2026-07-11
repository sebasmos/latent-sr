#!/bin/bash
# Run all tests

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "Running All Tests"
echo "========================================"

# Track failures
FAILED=0

# Test 1: Installation Files
echo -e "\n${YELLOW}[1/7] Testing Installation Files...${NC}"
if python3 tests/test_installation.py; then
    echo -e "${GREEN}✓ Installation files test passed${NC}"
else
    echo -e "${RED}✗ Installation files test failed${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 2: Environment Setup
echo -e "\n${YELLOW}[2/7] Testing Environment Setup...${NC}"
if python3 tests/test_environment.py; then
    echo -e "${GREEN}✓ Environment test passed${NC}"
else
    echo -e "${RED}✗ Environment test failed${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 3: Validation Framework
echo -e "\n${YELLOW}[3/7] Testing Validation Framework...${NC}"
if python3 tests/test_validation_framework.py; then
    echo -e "${GREEN}✓ Validation framework test passed${NC}"
else
    echo -e "${RED}✗ Validation framework test failed${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 4: Paper Validation Configuration
echo -e "\n${YELLOW}[4/7] Testing Paper Validation Config...${NC}"
if python3 tests/test_paper_validation.py; then
    echo -e "${GREEN}✓ Paper validation test passed${NC}"
else
    echo -e "${RED}✗ Paper validation test failed${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 5: Diffusion Architecture
echo -e "\n${YELLOW}[5/7] Testing Diffusion Architecture...${NC}"
if python3 tests/test_diffusion_architecture.py; then
    echo -e "${GREEN}✓ Diffusion architecture test passed${NC}"
else
    echo -e "${RED}✗ Diffusion architecture test failed${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 6: Reproducibility
echo -e "\n${YELLOW}[6/7] Testing Reproducibility & Requirements...${NC}"
if python3 tests/test_reproducibility.py; then
    echo -e "${GREEN}✓ Reproducibility test passed${NC}"
else
    echo -e "${RED}✗ Reproducibility test failed${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 7: Revision-Cycle Numbers
echo -e "\n${YELLOW}[7/7] Testing Revision-Cycle Numbers...${NC}"
if python3 tests/test_revision_validation.py; then
    echo -e "${GREEN}✓ Revision validation test passed${NC}"
else
    echo -e "${RED}✗ Revision validation test failed${NC}"
    FAILED=$((FAILED + 1))
fi

# Summary
echo ""
echo "========================================"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ $FAILED test(s) failed${NC}"
    exit 1
fi
