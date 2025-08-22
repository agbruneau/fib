package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"math/big"
	"math/bits"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"
)

// ----------------------------------------------------------------------------
// SECTION 1: DOMAIN LOGIC (Fibonacci Calculation Engine)
// ----------------------------------------------------------------------------

// --- Interface Definition ---

// Calculator interface defines the contract for Fibonacci algorithms.
// Refactoring: sync.Pool removed from the interface; it's an implementation detail.
type Calculator interface {
	Calculate(ctx context.Context, progressChan chan<- float64, n uint64) (*big.Int, error)
	Name() string
}

// --- Memory Management Utilities (Optimized) ---

// calculationTemps centralizes intermediate variables.
type calculationTemps struct {
	t1, t2, t3 *big.Int
}

// Optimization: Pool the entire calculationTemps struct instead of individual big.Ints.
// This reduces the number of allocations during calculation.
var tempsPool = sync.Pool{
	New: func() interface{} {
		return &calculationTemps{
			t1: new(big.Int),
			t2: new(big.Int),
			t3: new(big.Int),
		}
	},
}

// getTemps retrieves a set of initialized temporary variables from the pool.
func getTemps() *calculationTemps {
	return tempsPool.Get().(*calculationTemps)
}

// putTemps returns the temporary variables back to the pool for reuse.
func putTemps(temps *calculationTemps) {
	// Optimization: We do not explicitly reset the big.Ints (SetInt64(0)).
	// Since Calculate always overwrites the destination (t1, t2, t3) before reading,
	// resetting is unnecessary overhead and allows retaining the internal memory allocation.
	tempsPool.Put(temps)
}

// reportProgress sends the progress value to the channel non-blockingly.
func reportProgress(progressChan chan<- float64, progress float64) {
	if progressChan == nil {
		return
	}
	select {
	case progressChan <- progress:
	default:
		// Drop the update if the channel buffer is full.
	}
}

// --- Fast Doubling Implementation ---

type FastDoubling struct{}

func (fd *FastDoubling) Name() string {
	return "FastDoubling"
}

// Calculate computes the nth Fibonacci number using the Fast Doubling algorithm (O(log n)).
func (fd *FastDoubling) Calculate(ctx context.Context, progressChan chan<- float64, n uint64) (*big.Int, error) {

	// Handle base cases.
	if n == 0 {
		reportProgress(progressChan, 1.0)
		return big.NewInt(0), nil
	}
	if n <= 2 { // Optimization: Handle n=1 and n=2 quickly.
		reportProgress(progressChan, 1.0)
		return big.NewInt(1), nil
	}

	// Setup memory management using the optimized pool.
	temps := getTemps()
	defer putTemps(temps)

	// Initialize the state: F(0)=0, F(1)=1.
	f_k := big.NewInt(0)  // F(k)
	f_k1 := big.NewInt(1) // F(k+1)

	numBits := bits.Len64(n)
	// Optimization: Pre-calculate the inverse to use multiplication (faster) instead of division in the loop.
	invNumBits := 1.0 / float64(numBits)

	// Iterate from MSB to LSB.
	for i := numBits - 1; i >= 0; i-- {

		// Check for cancellation signal periodically.
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("calculation canceled: %w", ctx.Err())
		default:
			// Proceed with calculation.
		}

		// Calculate progression.
		if progressChan != nil && i < numBits-1 {
			progress := float64(numBits-1-i) * invNumBits
			reportProgress(progressChan, progress)
		}

		// --- Doubling Step (k -> 2k) ---

		// 1. F(2k) = F(k) * [2*F(k+1) - F(k)]
		temps.t1.Lsh(f_k1, 1)       // t1 = 2 * F(k+1)
		temps.t2.Sub(temps.t1, f_k) // t2 = [2*F(k+1) - F(k)]
		temps.t3.Mul(f_k, temps.t2) // t3 = F(2k)

		// 2. F(2k+1) = F(k+1)^2 + F(k)^2
		temps.t1.Mul(f_k1, f_k1)     // t1 = F(k+1)^2
		temps.t2.Mul(f_k, f_k)       // t2 = F(k)^2
		f_k1.Add(temps.t1, temps.t2) // F(2k+1)

		f_k.Set(temps.t3) // Update F(k) to F(2k)

		// --- Addition Step (k -> k+1 if necessary) ---
		if (n>>uint(i))&1 == 1 {
			temps.t1.Set(f_k1)
			f_k1.Add(f_k1, f_k)
			f_k.Set(temps.t1)
		}
	}

	reportProgress(progressChan, 1.0)
	return f_k, nil
}

// ----------------------------------------------------------------------------
// SECTION 2: APPLICATION LOGIC (CLI Interface)
// ----------------------------------------------------------------------------

// Define exit codes.
const (
	ExitSuccess       = 0
	ExitErrorGeneric  = 1
	ExitErrorTimeout  = 2
	ExitErrorCanceled = 130 // Standard code for Ctrl+C
)

// AppConfig centralizes the configuration.
type AppConfig struct {
	N       uint64
	Verbose bool
	Timeout time.Duration
}

func main() {
	// 1. Configuration des arguments CLI
	// We define flags here. flag.Parse() must occur before use.
	nFlag := flag.Uint64("n", 100000000, "L'indice 'n' de la séquence de Fibonacci à calculer.")
	verboseFlag := flag.Bool("v", false, "Affiche le résultat complet.")
	timeoutFlag := flag.Duration("timeout", 5*time.Minute, "Délai maximum pour le calcul (ex: 30s, 1m).")
	flag.Parse()

	config := AppConfig{
		N:       *nFlag,
		Verbose: *verboseFlag,
		Timeout: *timeoutFlag,
	}

	// 2. Execution de l'application (Refactoring: main calls run)
	// We pass os.Stdout for dependency injection, enabling testing.
	exitCode := run(context.Background(), config, os.Stdout)
	os.Exit(exitCode)
}

// run orchestrates the application logic. Separated from main() for testability.
func run(ctx context.Context, config AppConfig, out io.Writer) int {
	fmt.Fprintf(out, "Calcul de F(%d) avec l'algorithme Fast Doubling...\n", config.N)
	fmt.Fprintf(out, "Timeout défini à %s.\n", config.Timeout)

	// 1. Sélection de la Stratégie
	var calculator Calculator = &FastDoubling{}

	// 2. Configuration du Contexte et de l'Annulation
	ctx, cancelTimeout := context.WithTimeout(ctx, config.Timeout)
	defer cancelTimeout()

	// Gestion des signaux (Ctrl+C)
	ctx, stopSignals := signal.NotifyContext(ctx, syscall.SIGINT, syscall.SIGTERM)
	defer stopSignals()

	// 3. Configuration du Reporting de Progression
	progressChan := make(chan float64, 100)
	var wg sync.WaitGroup

	wg.Add(1)
	go displayProgress(&wg, progressChan, out)

	// 4. Exécution du Calcul
	startTime := time.Now()
	// The pool is now managed internally by the Calculator implementation.
	result, err := calculator.Calculate(ctx, progressChan, config.N)

	// Signal completion and wait for the progress display.
	close(progressChan)
	wg.Wait()

	duration := time.Since(startTime)

	// 5. Gestion des Résultats et Erreurs
	fmt.Fprintln(out, "\n--- Résultats ---")
	if err != nil {
		return handleCalculationError(err, duration, config.Timeout, out)
	}

	// 6. Affichage du Résultat
	displayResult(result, config.N, duration, config.Verbose, out)
	return ExitSuccess
}

// handleCalculationError determines the cause of the error and reports it.
func handleCalculationError(err error, duration time.Duration, timeout time.Duration, out io.Writer) int {
	// Refactoring: Use errors.Is for robust error checking.
	if errors.Is(err, context.DeadlineExceeded) {
		fmt.Fprintf(out, "Statut : Échec. Le calcul a dépassé le délai imparti (%s) après %s.\n", timeout, duration)
		return ExitErrorTimeout
	} else if errors.Is(err, context.Canceled) {
		fmt.Fprintf(out, "Statut : Annulé (signal reçu ou Ctrl+C) après %s.\n", duration)
		return ExitErrorCanceled
	} else {
		fmt.Fprintf(out, "Statut : Échec. Erreur interne : %v\n", err)
		return ExitErrorGeneric
	}
}

// displayResult formats and prints the final calculation result.
func displayResult(result *big.Int, n uint64, duration time.Duration, verbose bool, out io.Writer) {
	fmt.Fprintf(out, "Statut : Succès\n")
	fmt.Fprintf(out, "Durée d'exécution : %s\n", duration)
	fmt.Fprintf(out, "Taille du résultat : %d bits.\n", result.BitLen())

	resultStr := result.String()
	numDigits := len(resultStr)
	fmt.Fprintf(out, "Nombre de chiffres décimaux : %d\n", numDigits)

	const truncationLimit = 50
	const displayEdges = 20

	if verbose {
		fmt.Fprintf(out, "F(%d) = %s\n", n, resultStr)
	} else if numDigits > truncationLimit {
		// Affichage tronqué
		fmt.Fprintf(out, "F(%d) (tronqué) = %s...%s\n", n, resultStr[:displayEdges], resultStr[numDigits-displayEdges:])
	} else {
		fmt.Fprintf(out, "F(%d) = %s\n", n, resultStr)
	}
}

// displayProgress reads the progress channel and updates the display periodically.
func displayProgress(wg *sync.WaitGroup, progressChan <-chan float64, out io.Writer) {
	defer wg.Done()
	// Ticker (100ms = 10Hz) to limit refresh rate.
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	lastProgress := 0.0

	// Helper function to print the current progress bar
	printBar := func(progress float64) {
		// \r (Carriage Return) moves the cursor to the beginning of the line.
		fmt.Fprintf(out, "\rProgression : %6.2f%% [%-30s]", progress*100, progressBar(progress, 30))
	}

	for {
		select {
		case p, ok := <-progressChan:
			if !ok {
				// Channel closed. Ensure 100% is displayed.
				if lastProgress != 1.0 {
					printBar(1.0)
				}
				fmt.Fprintln(out) // New line after the final progress bar.
				return
			}
			lastProgress = p
		case <-ticker.C:
			// Update display at regular interval.
			printBar(lastProgress)
		}
	}
}

// progressBar generates a CLI progress bar string.
// Optimization: Using strings.Builder for efficient string construction.
func progressBar(progress float64, length int) string {
	// Clamping progress between 0.0 and 1.0
	if progress > 1.0 {
		progress = 1.0
	} else if progress < 0.0 {
		progress = 0.0
	}

	count := int(progress * float64(length))

	var builder strings.Builder
	builder.Grow(length) // Pre-allocate capacity

	for i := 0; i < length; i++ {
		if i < count {
			builder.WriteRune('■')
		} else {
			builder.WriteRune(' ')
		}
	}
	return builder.String()
}
