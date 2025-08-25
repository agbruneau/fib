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

func TestOptimizedFastDoubling_Calculate(t *testing.T) {
	// Test cases covering various scenarios.
	testCases := []struct {
		name     string
		n        uint64
		expected *big.Int
	}{
		{"F(0)", 0, big.NewInt(0)},
		{"F(1)", 1, big.NewInt(1)},
		{"F(2)", 2, big.NewInt(1)},
		{"F(10)", 10, big.NewInt(55)},
		{"F(20)", 20, big.NewInt(6765)},
		// Test the edge of uint64 calculation (fast path).
		{"F(92)", 92, bigInt("7540113804746346429")},
		{"F(93)", MaxFibUint64, bigInt("12200160415121876738")},
		// Test the transition to big.Int calculation (slow path).
		{"F(94)", 94, bigInt("19740274219868223167")},
		{"F(100)", 100, bigInt("354224848179261915075")},
		{"F(200)", 200, bigInt("280571172992510140037611932413038677189525")},
		{"F(250)", 250, bigInt("7896325826131730509282738943634332893686268675876375")},
	}

	calculator := &OptimizedFastDoubling{}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Use a background context as we are not testing cancellation here.
			ctx := context.Background()

			// We don't need to test the progress channel here, so we pass nil.
			result, err := calculator.Calculate(ctx, nil, tc.n)

			if err != nil {
				t.Fatalf("Calculate() returned an unexpected error: %v", err)
			}

			// Cmp returns -1, 0, or 1. We expect 0 for equality.
			if result.Cmp(tc.expected) != 0 {
				t.Errorf("Calculate(%d) = %s; want %s", tc.n, result.String(), tc.expected.String())
			}
		})
	}
}

// Benchmark the two calculation paths.
func BenchmarkCalculate(b *testing.B) {
	calculator := &OptimizedFastDoubling{}
	ctx := context.Background()

	b.Run("SmallN_FastPath", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			calculator.Calculate(ctx, nil, 92)
		}
	})

	b.Run("LargeN_BigIntPath", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			calculator.Calculate(ctx, nil, 100000) // A moderately large number
		}
	})
}

// --- Tests for Application Logic (run function) ---

func TestRun_Success(t *testing.T) {
	t.Run("NormalOutput", func(t *testing.T) {
		var out bytes.Buffer
		config := AppConfig{N: 20, Verbose: false, Timeout: 1 * time.Minute}

		exitCode := run(context.Background(), config, &out)

		if exitCode != ExitSuccess {
			t.Errorf("Expected exit code %d, got %d", ExitSuccess, exitCode)
		}

		output := out.String()
		if !strings.Contains(output, "F(20) = 6765") {
			t.Errorf("Expected output to contain 'F(20) = 6765', but it didn't. Got: %s", output)
		}
	})

	t.Run("VerboseOutput", func(t *testing.T) {
		var out bytes.Buffer
		// For n=250, the output should be full with verbose.
		config := AppConfig{N: 250, Verbose: true, Timeout: 1 * time.Minute}

		run(context.Background(), config, &out)

		output := out.String()
		expectedResult := "F(250) = 7896325826131730509282738943634332893686268675876375"
		if !strings.Contains(output, expectedResult) {
			t.Errorf("Expected verbose output to contain '%s', but it didn't. Got: %s", expectedResult, output)
		}
	})

	t.Run("TruncatedOutput", func(t *testing.T) {
		var out bytes.Buffer
		// For n=250, the output should be truncated as it has > 50 digits.
		config := AppConfig{N: 250, Verbose: false, Timeout: 1 * time.Minute}

		run(context.Background(), config, &out)

		output := out.String()
		// Check for the truncated format "start...end"
		expectedTrunc := "F(250) (tronqué) = 78963258261317305092...32893686268675876375"
		if !strings.Contains(output, expectedTrunc) {
			t.Errorf("Expected output to be truncated. Expected to contain '%s'. Got: %s", expectedTrunc, output)
		}
	})
}

func TestRun_Errors(t *testing.T) {
	t.Run("Timeout", func(t *testing.T) {
		var out bytes.Buffer
		// Use a large N that will surely take more than 1 millisecond.
		// Use a very short timeout to trigger the error quickly.
		config := AppConfig{N: 10000000, Timeout: 1 * time.Millisecond}

		exitCode := run(context.Background(), config, &out)

		if exitCode != ExitErrorTimeout {
			t.Errorf("Expected exit code %d for timeout, got %d", ExitErrorTimeout, exitCode)
		}

		output := out.String()
		if !strings.Contains(output, "Le calcul a dépassé le délai imparti") {
			t.Errorf("Expected output to contain timeout message, but it didn't. Got: %s", output)
		}
	})

	t.Run("Cancellation", func(t *testing.T) {
		var out bytes.Buffer
		config := AppConfig{N: 10000000, Timeout: 1 * time.Minute}

		// Create a context that is immediately cancelled.
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		exitCode := run(ctx, config, &out)

		if exitCode != ExitErrorCanceled {
			t.Errorf("Expected exit code %d for cancellation, got %d", ExitErrorCanceled, exitCode)
		}

		output := out.String()
		if !strings.Contains(output, "Statut : Annulé") {
			t.Errorf("Expected output to contain cancellation message, but it didn't. Got: %s", output)
		}
	})
}
