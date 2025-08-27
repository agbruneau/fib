package main

import (
	"bytes"
	"context"
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

// testCases shared between both algorithm tests.
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

func TestOptimizedFastDoubling_Calculate(t *testing.T) {
	calculator := calculatorRegistry["fast"]
	if calculator == nil {
		t.Fatal("calculator 'fast' not found in registry")
	}

	for _, tc := range fibTestCases {
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
}

func TestMatrixExponentiation_Calculate(t *testing.T) {
	calculator := calculatorRegistry["matrix"]
	if calculator == nil {
		t.Fatal("calculator 'matrix' not found in registry")
	}

	for _, tc := range fibTestCases {
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
}

func TestRun_Success(t *testing.T) {
	t.Run("NormalOutput", func(t *testing.T) {
		var out bytes.Buffer
		config := AppConfig{N: 20, Verbose: false, Timeout: 1 * time.Minute, Algo: "fast"}
		exitCode := run(context.Background(), config, &out)
		if exitCode != ExitSuccess {
			t.Errorf("Expected exit code %d, got %d", ExitSuccess, exitCode)
		}
		if !strings.Contains(out.String(), "F(20) = 6765") {
			t.Errorf("Expected output to contain 'F(20) = 6765', but it didn't. Got: %s", out.String())
		}
	})

	t.Run("VerboseOutput", func(t *testing.T) {
		var out bytes.Buffer
		config := AppConfig{N: 250, Verbose: true, Timeout: 1 * time.Minute, Algo: "fast"}
		run(context.Background(), config, &out)
		expectedResult := "F(250) = 7896325826131730509282738943634332893686268675876375"
		if !strings.Contains(out.String(), expectedResult) {
			t.Errorf("Expected verbose output to contain '%s', but it didn't. Got: %s", expectedResult, out.String())
		}
	})

	t.Run("TruncatedOutput", func(t *testing.T) {
		var out bytes.Buffer
		config := AppConfig{N: 250, Verbose: false, Timeout: 1 * time.Minute, Algo: "fast"}
		run(context.Background(), config, &out)
		expectedTrunc := "F(250) (tronqué) = 78963258261317305092...32893686268675876375"
		if !strings.Contains(out.String(), expectedTrunc) {
			t.Errorf("Expected output to be truncated. Expected to contain '%s'. Got: %s", expectedTrunc, out.String())
		}
	})
}

func TestRun_Errors(t *testing.T) {
	t.Run("Timeout", func(t *testing.T) {
		var out bytes.Buffer
		// Use a fast algorithm so timeout is reliable
		config := AppConfig{N: 10000000, Timeout: 1 * time.Millisecond, Algo: "fast"}
		exitCode := run(context.Background(), config, &out)
		if exitCode != ExitErrorTimeout {
			t.Errorf("Expected exit code %d for timeout, got %d", ExitErrorTimeout, exitCode)
		}
		if !strings.Contains(out.String(), "Le calcul a dépassé le délai imparti") {
			t.Errorf("Expected output to contain timeout message, but it didn't. Got: %s", out.String())
		}
	})

	t.Run("Cancellation", func(t *testing.T) {
		var out bytes.Buffer
		config := AppConfig{N: 10000000, Timeout: 1 * time.Minute, Algo: "fast"}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		exitCode := run(ctx, config, &out)
		if exitCode != ExitErrorCanceled {
			t.Errorf("Expected exit code %d for cancellation, got %d", ExitErrorCanceled, exitCode)
		}
		if !strings.Contains(out.String(), "Statut : Annulé") {
			t.Errorf("Expected output to contain cancellation message, but it didn't. Got: %s", out.String())
		}
	})
}
