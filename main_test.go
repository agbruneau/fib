package main

import (
	"bytes"
	"context"
	"fmt"
	"math/big"
	"strings"
	"testing"
	"time"
)

// bigInt is a helper function to create a new big.Int from a string.
func bigInt(s string) *big.Int {
	n, ok := new(big.Int).SetString(s, 10)
	if !ok {
		panic("failed to create big.Int from string: " + s)
	}
	return n
}

// TestCalculators is a table-driven test that covers all registered Fibonacci calculators.
func TestCalculators(t *testing.T) {
	// Shared test cases for all Fibonacci calculation algorithms.
	var fibTestCases = []struct {
		name     string
		n        uint64
		expected *big.Int
	}{
		{"F(0)", 0, big.NewInt(0)},
		{"F(1)", 1, big.NewInt(1)},
		{"F(10)", 10, big.NewInt(55)},
		{"F(93)", MaxFibUint64, bigInt("12200160415121876738")},
		{"F(94)", 94, bigInt("19740274219868223167")},
		{"F(250)", 250, bigInt("7896325826131730509282738943634332893686268675876375")},
	}

	// Iterate over all registered calculators.
	for name, calculator := range calculatorRegistry {
		t.Run(fmt.Sprintf("Calculator_%s", name), func(t *testing.T) {
			t.Parallel() // Run calculators in parallel.
			for _, tc := range fibTestCases {
				// Run each specific test case as a sub-test.
				t.Run(tc.name, func(t *testing.T) {
					ctx := context.Background()
					result, err := calculator.Calculate(ctx, nil, tc.n)
					if err != nil {
						t.Fatalf("Calculate() returned an unexpected error: %v", err)
					}
					if result.Cmp(tc.expected) != 0 {
						t.Errorf("Calculate(%d) = %s; want %s", tc.n, result.String(), tc.expected.String())
					}
				})
			}
		})
	}
}

// TestRun orchestrates tests for the `run` function, covering various scenarios.
func TestRun(t *testing.T) {
	// Base configuration, can be overridden by specific test cases.
	baseConfig := AppConfig{
		Timeout: 1 * time.Minute,
		Algo:    "fast", // Use a fast algorithm for most tests.
	}

	testCases := []struct {
		name                 string
		config               AppConfig
		contextSetup         func() (context.Context, context.CancelFunc)
		expectedExitCode     int
		expectedOutput       []string
		unexpectedOutput     []string
		assertOutputContains bool
	}{
		{
			name:   "Success/NormalOutput",
			config: AppConfig{N: 20, Verbose: false},
			expectedExitCode: ExitSuccess,
			expectedOutput:   []string{"F(20) = 6765"},
		},
		{
			name:   "Success/VerboseOutput",
			config: AppConfig{N: 250, Verbose: true},
			expectedExitCode: ExitSuccess,
			expectedOutput: []string{
				"F(250) = 7896325826131730509282738943634332893686268675876375",
			},
		},
		{
			name:   "Success/TruncatedOutput",
			config: AppConfig{N: 250, Verbose: false},
			expectedExitCode: ExitSuccess,
			expectedOutput: []string{
				"F(250) (tronqué) = 78963258261317305092...32893686268675876375",
			},
		},
		{
			name:   "Error/Timeout",
			config: AppConfig{N: 100000000, Timeout: 1 * time.Millisecond},
			expectedExitCode: ExitErrorTimeout,
			expectedOutput:   []string{"Le calcul a dépassé le délai imparti"},
		},
		{
			name:   "Error/Cancellation",
			config: AppConfig{N: 100000000},
			contextSetup: func() (context.Context, context.CancelFunc) {
				ctx, cancel := context.WithCancel(context.Background())
				cancel() // Immediately cancel the context.
				return ctx, cancel
			},
			expectedExitCode: ExitErrorCanceled,
			expectedOutput:   []string{"Statut : Annulé"},
		},
		{
			name:             "Error/UnknownAlgorithm",
			config:           AppConfig{Algo: "unknown"},
			expectedExitCode: ExitErrorGeneric,
			expectedOutput:   []string{"Erreur : algorithme 'unknown' inconnu"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var out bytes.Buffer
			// Apply base config defaults if not set in test case
			finalConfig := baseConfig
			if tc.config.N != 0 {
				finalConfig.N = tc.config.N
			}
			finalConfig.Verbose = tc.config.Verbose
			if tc.config.Timeout != 0 {
				finalConfig.Timeout = tc.config.Timeout
			}
			if tc.config.Algo != "" {
				finalConfig.Algo = tc.config.Algo
			}

			ctx := context.Background()
			if tc.contextSetup != nil {
				var cancel context.CancelFunc
				ctx, cancel = tc.contextSetup()
				defer cancel()
			}

			exitCode := run(ctx, finalConfig, &out)

			if exitCode != tc.expectedExitCode {
				t.Errorf("Expected exit code %d, got %d. Output:\n%s", tc.expectedExitCode, exitCode, out.String())
			}

			outputStr := out.String()
			for _, expected := range tc.expectedOutput {
				if !strings.Contains(outputStr, expected) {
					t.Errorf("Expected output to contain '%s', but it didn't.\nFull output:\n%s", expected, outputStr)
				}
			}
			for _, unexpected := range tc.unexpectedOutput {
				if strings.Contains(outputStr, unexpected) {
					t.Errorf("Expected output to NOT contain '%s', but it did.\nFull output:\n%s", unexpected, outputStr)
				}
			}
		})
	}
}
