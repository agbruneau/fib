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
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"
)

// ----------------------------------------------------------------------------
// SECTION 1: DOMAIN LOGIC (Fibonacci Calculation Engine - Highly Optimized)
// ----------------------------------------------------------------------------

// MaxFibUint64 is the largest index N such that F(N) fits in a uint64.
const MaxFibUint64 = 93

// parallelThreshold defines the minimum bit length for parallel squaring.
const parallelThreshold = 2048

type Calculator interface {
	Calculate(ctx context.Context, progressChan chan<- float64, n uint64) (*big.Int, error)
	Name() string
}

// --- Memory Management (O2: Zero Allocation Strategy) ---

// calculationState holds ALL big.Int variables needed for the calculation.
type calculationState struct {
	// State variables
	f_k  *big.Int // F(k)
	f_k1 *big.Int // F(k+1)
	// Temporary variables (t4 added for safe parallel operations O3)
	t1, t2, t3, t4 *big.Int
}

// statePool pools the entire calculation state to achieve near-zero allocation.
var statePool = sync.Pool{
	New: func() interface{} {
		return &calculationState{
			f_k:  new(big.Int),
			f_k1: new(big.Int),
			t1:   new(big.Int),
			t2:   new(big.Int),
			t3:   new(big.Int),
			t4:   new(big.Int),
		}
	},
}

// getState retrieves and initializes the state from the pool.
func getState() *calculationState {
	s := statePool.Get().(*calculationState)
	// Initialize state: F(0)=0, F(1)=1.
	s.f_k.SetInt64(0)
	s.f_k1.SetInt64(1)
	// Temporaries do not need resetting as they are always overwritten before read.
	return s
}

// putState returns the state to the pool.
func putState(s *calculationState) {
	statePool.Put(s)
}

// reportProgress sends updates non-blockingly.
func reportProgress(progressChan chan<- float64, progress float64) {
	if progressChan == nil {
		return
	}
	select {
	case progressChan <- progress:
	default:
	}
}

// --- Optimized Implementation ---

type OptimizedFastDoubling struct{}

func (fd *OptimizedFastDoubling) Name() string {
	return "OptimizedFastDoubling (Parallel+ZeroAlloc+FastPath)"
}

// O1: Fast Path for small N (N <= 93) using native uint64.
func (fd *OptimizedFastDoubling) calculateSmall(n uint64) *big.Int {
	if n == 0 {
		return big.NewInt(0)
	}
	var a, b uint64 = 0, 1
	// Iterative calculation using native arithmetic.
	for i := uint64(1); i < n; i++ {
		a, b = b, a+b
	}
	return new(big.Int).SetUint64(b)
}

// Calculate computes the nth Fibonacci number.
func (fd *OptimizedFastDoubling) Calculate(ctx context.Context, progressChan chan<- float64, n uint64) (*big.Int, error) {

	// 1. Handle Small N (O1)
	if n <= MaxFibUint64 {
		reportProgress(progressChan, 1.0)
		return fd.calculateSmall(n), nil
	}

	// 2. Setup for Large N (O2: Zero Allocation)
	// Get the entire state from the pool. f_k and f_k1 are initialized.
	s := getState()
	defer putState(s)

	numBits := bits.Len64(n)
	// Micro-optimization: Use multiplication by inverse instead of division in the loop.
	invNumBits := 1.0 / float64(numBits)

	// Setup for parallel operations (O3)
	var wg sync.WaitGroup
	useParallel := runtime.NumCPU() > 1

	// 3. Main loop (MSB to LSB)
	for i := numBits - 1; i >= 0; i-- {

		// Check for cancellation periodically.
		if ctx.Err() != nil {
			// Use select {} for non-blocking check if preferred, but direct check is fine here.
			return nil, fmt.Errorf("calculation canceled: %w", ctx.Err())
		}

		// Report progression.
		if progressChan != nil && i < numBits-1 {
			progress := float64(numBits-1-i) * invNumBits
			reportProgress(progressChan, progress)
		}

		// --- Doubling Step (k -> 2k) ---

		// 1. F(2k) = F(k) * [2*F(k+1) - F(k)] (Sequential part)
		s.t1.Lsh(s.f_k1, 1)   // t1 = 2 * F(k+1)
		s.t2.Sub(s.t1, s.f_k) // t2 = [2*F(k+1) - F(k)]
		s.t3.Mul(s.f_k, s.t2) // t3 = F(2k) (MULTIPLICATION 1)

		// 2. F(2k+1) = F(k+1)^2 + F(k)^2 (O3: Parallelizable part)

		// Check threshold and CPU availability.
		if useParallel && s.f_k1.BitLen() > parallelThreshold {
			// Execute the two independent squarings concurrently.
			wg.Add(2)

			// Goroutine 1: Calculate F(k+1)^2 -> t1
			go func(dest, src *big.Int) {
				defer wg.Done()
				dest.Mul(src, src) // MULTIPLICATION 2
			}(s.t1, s.f_k1)

			// Goroutine 2: Calculate F(k)^2 -> t4 (Using t4 to avoid race with t1)
			go func(dest, src *big.Int) {
				defer wg.Done()
				dest.Mul(src, src) // MULTIPLICATION 3
			}(s.t4, s.f_k)

			wg.Wait()
			s.f_k1.Add(s.t1, s.t4) // Combine results

		} else {
			// Sequential execution for smaller numbers or single core.
			s.t1.Mul(s.f_k1, s.f_k1) // MULTIPLICATION 2
			s.t4.Mul(s.f_k, s.f_k)   // MULTIPLICATION 3
			s.f_k1.Add(s.t1, s.t4)
		}

		s.f_k.Set(s.t3) // Update F(k) to F(2k)

		// --- Addition Step (k -> k+1 if necessary) ---
		if (n>>uint(i))&1 == 1 {
			s.t1.Set(s.f_k1)
			s.f_k1.Add(s.f_k1, s.f_k)
			s.f_k.Set(s.t1)
		}
	}

	reportProgress(progressChan, 1.0)

	// CRITICAL (O2): Return a COPY of the result.
	// s.f_k will be returned to the pool via defer and must not be used by the caller.
	return new(big.Int).Set(s.f_k), nil
}

// ----------------------------------------------------------------------------
// SECTION 2: APPLICATION LOGIC (CLI Interface)
// (The CLI remains largely the same as the refactored version, ensuring robustness)
// ----------------------------------------------------------------------------

const (
	ExitSuccess       = 0
	ExitErrorGeneric  = 1
	ExitErrorTimeout  = 2
	ExitErrorCanceled = 130
)

type AppConfig struct {
	N       uint64
	Verbose bool
	Timeout time.Duration
}

func main() {
	// Configuration CLI
	// Increased default N to better demonstrate optimizations.
	nFlag := flag.Uint64("n", 100000000, "L'indice 'n' de la séquence de Fibonacci à calculer.")
	verboseFlag := flag.Bool("v", false, "Affiche le résultat complet.")
	timeoutFlag := flag.Duration("timeout", 5*time.Minute, "Délai maximum (ex: 30s, 1m).")
	flag.Parse()

	config := AppConfig{
		N:       *nFlag,
		Verbose: *verboseFlag,
		Timeout: *timeoutFlag,
	}

	// Execution de l'application (Refactorisé pour testabilité)
	exitCode := run(context.Background(), config, os.Stdout)
	os.Exit(exitCode)
}

func run(ctx context.Context, config AppConfig, out io.Writer) int {
	var calculator Calculator = &OptimizedFastDoubling{}

	fmt.Fprintf(out, "Calcul de F(%d)...\n", config.N)
	fmt.Fprintf(out, "Algorithme : %s\n", calculator.Name())
	fmt.Fprintf(out, "Nombre de cœurs CPU disponibles : %d\n", runtime.NumCPU())
	fmt.Fprintf(out, "Timeout défini à %s.\n", config.Timeout)

	// Configuration du Contexte et Annulation
	ctx, cancelTimeout := context.WithTimeout(ctx, config.Timeout)
	defer cancelTimeout()
	ctx, stopSignals := signal.NotifyContext(ctx, syscall.SIGINT, syscall.SIGTERM)
	defer stopSignals()

	// Reporting de Progression
	progressChan := make(chan float64, 100)
	var wg sync.WaitGroup
	wg.Add(1)
	go displayProgress(&wg, progressChan, out)

	// Exécution du Calcul
	startTime := time.Now()
	result, err := calculator.Calculate(ctx, progressChan, config.N)

	close(progressChan)
	wg.Wait()
	duration := time.Since(startTime)

	// Gestion des Résultats et Erreurs
	fmt.Fprintln(out, "\n--- Résultats ---")
	if err != nil {
		return handleCalculationError(err, duration, config.Timeout, out)
	}

	displayResult(result, config.N, duration, config.Verbose, out)
	return ExitSuccess
}

// handleCalculationError uses errors.Is for robust context error detection.
func handleCalculationError(err error, duration time.Duration, timeout time.Duration, out io.Writer) int {
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
		fmt.Fprintf(out, "F(%d) (tronqué) = %s...%s\n", n, resultStr[:displayEdges], resultStr[numDigits-displayEdges:])
	} else {
		fmt.Fprintf(out, "F(%d) = %s\n", n, resultStr)
	}
}

func displayProgress(wg *sync.WaitGroup, progressChan <-chan float64, out io.Writer) {
	defer wg.Done()
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	lastProgress := 0.0

	printBar := func(progress float64) {
		// \r moves the cursor to the beginning of the line.
		fmt.Fprintf(out, "\rProgression : %6.2f%% [%-30s]", progress*100, progressBar(progress, 30))
	}

	for {
		select {
		case p, ok := <-progressChan:
			if !ok {
				// Channel closed, ensure 100% is displayed.
				printBar(1.0)
				fmt.Fprintln(out)
				return
			}
			lastProgress = p
		case <-ticker.C:
			printBar(lastProgress)
		}
	}
}

// progressBar optimized using strings.Builder.
func progressBar(progress float64, length int) string {
	if progress > 1.0 {
		progress = 1.0
	} else if progress < 0.0 {
		progress = 0.0
	}

	count := int(progress * float64(length))

	var builder strings.Builder
	builder.Grow(length)

	for i := 0; i < length; i++ {
		if i < count {
			builder.WriteRune('■')
		} else {
			builder.WriteRune(' ')
		}
	}
	return builder.String()
}
