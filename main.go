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
// SECTION 1: MOTEURS DE CALCUL FIBONACCI (Logique principale, hautement optimisée)
// ----------------------------------------------------------------------------

const (
	// MaxFibUint64 est l'index N le plus élevé pour lequel F(N) peut être stocké
	// dans un entier non signé de 64 bits (uint64). Au-delà de F(93), le résultat
	// dépasse la capacité de ce type de données.
	MaxFibUint64 = 93
	// parallelThreshold définit la taille minimale (en bits) d'un nombre à partir de
	// laquelle il devient plus performant d'effectuer les multiplications en parallèle
	// sur plusieurs cœurs de processeur.
	parallelThreshold = 2048
)

// Calculator est une interface décrivant un objet capable de calculer un nombre de Fibonacci.
type Calculator interface {
	Calculate(ctx context.Context, progressChan chan<- float64, n uint64) (*big.Int, error)
	Name() string
}

// reportProgress envoie des mises à jour de manière non bloquante.
func reportProgress(progressChan chan<- float64, progress float64) {
	if progressChan == nil {
		return
	}
	select {
	case progressChan <- progress:
	default:
	}
}

// calculateSmall est une optimisation partagée pour les petits nombres (n <= 93).
func calculateSmall(n uint64) *big.Int {
	if n == 0 {
		return big.NewInt(0)
	}
	var a, b uint64 = 0, 1
	for i := uint64(1); i < n; i++ {
		a, b = b, a+b
	}
	return new(big.Int).SetUint64(b)
}

// ----------------------------------------------------------------------------
// IMPLÉMENTATION 1: ALGORITHME "OPTIMIZED FAST DOUBLING"
// ----------------------------------------------------------------------------

type calculationState struct {
	f_k, f_k1, t1, t2, t3, t4 *big.Int
}

var statePool = sync.Pool{
	New: func() interface{} {
		return &calculationState{
			f_k: new(big.Int), f_k1: new(big.Int),
			t1: new(big.Int), t2: new(big.Int),
			t3: new(big.Int), t4: new(big.Int),
		}
	},
}

func getState() *calculationState {
	s := statePool.Get().(*calculationState)
	s.f_k.SetInt64(0)
	s.f_k1.SetInt64(1)
	return s
}

func putState(s *calculationState) {
	statePool.Put(s)
}

type OptimizedFastDoubling struct{}

func (fd *OptimizedFastDoubling) Name() string {
	return "OptimizedFastDoubling (3-Way-Parallel+ZeroAlloc+FastPath)"
}

func (fd *OptimizedFastDoubling) Calculate(ctx context.Context, progressChan chan<- float64, n uint64) (*big.Int, error) {
	if n <= MaxFibUint64 {
		reportProgress(progressChan, 1.0)
		return calculateSmall(n), nil
	}

	s := getState()
	defer putState(s)

	numBits := bits.Len64(n)
	invNumBits := 1.0 / float64(numBits)
	var wg sync.WaitGroup
	useParallel := runtime.NumCPU() > 1

	for i := numBits - 1; i >= 0; i-- {
		if ctx.Err() != nil {
			return nil, fmt.Errorf("calculation canceled: %w", ctx.Err())
		}
		if progressChan != nil && i < numBits-1 {
			reportProgress(progressChan, float64(numBits-1-i)*invNumBits)
		}

		// Étape de "Doubling"
		s.t2.Lsh(s.f_k1, 1)
		s.t2.Sub(s.t2, s.f_k)

		if useParallel && s.f_k1.BitLen() > parallelThreshold {
			wg.Add(3)
			go func(dest, src1, src2 *big.Int) { defer wg.Done(); dest.Mul(src1, src2) }(s.t3, s.f_k, s.t2)
			go func(dest, src *big.Int) { defer wg.Done(); dest.Mul(src, src) }(s.t1, s.f_k1)
			go func(dest, src *big.Int) { defer wg.Done(); dest.Mul(src, src) }(s.t4, s.f_k)
			wg.Wait()
			s.f_k.Set(s.t3)
			s.f_k1.Add(s.t1, s.t4)
		} else {
			s.t3.Mul(s.f_k, s.t2)
			s.t1.Mul(s.f_k1, s.f_k1)
			s.t4.Mul(s.f_k, s.f_k)
			s.f_k.Set(s.t3)
			s.f_k1.Add(s.t1, s.t4)
		}

		// Étape d'"Addition"
		if (n>>uint(i))&1 == 1 {
			s.t1.Set(s.f_k1)
			s.f_k1.Add(s.f_k1, s.f_k)
			s.f_k.Set(s.t1)
		}
	}

	reportProgress(progressChan, 1.0)
	return new(big.Int).Set(s.f_k), nil
}

// ----------------------------------------------------------------------------
// IMPLÉMENTATION 2: ALGORITHME D'EXPONENTIATION MATRICIELLE
// ----------------------------------------------------------------------------

type matrix struct {
	a, b, c, d *big.Int
}

type matrixState struct {
	res, p                         *matrix
	t1, t2, t3, t4, t5, t6, t7, t8 *big.Int
}

var matrixStatePool = sync.Pool{
	New: func() interface{} {
		return &matrixState{
			res: &matrix{new(big.Int), new(big.Int), new(big.Int), new(big.Int)},
			p:   &matrix{new(big.Int), new(big.Int), new(big.Int), new(big.Int)},
			t1:  new(big.Int), t2: new(big.Int), t3: new(big.Int), t4: new(big.Int),
			t5: new(big.Int), t6: new(big.Int), t7: new(big.Int), t8: new(big.Int),
		}
	},
}

func getMatrixState() *matrixState {
	s := matrixStatePool.Get().(*matrixState)
	s.res.a.SetInt64(1)
	s.res.b.SetInt64(0)
	s.res.c.SetInt64(0)
	s.res.d.SetInt64(1)
	s.p.a.SetInt64(1)
	s.p.b.SetInt64(1)
	s.p.c.SetInt64(1)
	s.p.d.SetInt64(0)
	return s
}

func putMatrixState(s *matrixState) {
	matrixStatePool.Put(s)
}

type MatrixExponentiation struct{}

func (me *MatrixExponentiation) Name() string {
	return "MatrixExponentiation (Parallel+ZeroAlloc+FastPath)"
}

func multiplyMatrices(dest, m1, m2 *matrix, s *matrixState, useParallel bool) {
	var wg sync.WaitGroup
	if useParallel && m1.a.BitLen() > parallelThreshold {
		wg.Add(8)
		go func() { defer wg.Done(); s.t1.Mul(m1.a, m2.a) }()
		go func() { defer wg.Done(); s.t2.Mul(m1.b, m2.c) }()
		go func() { defer wg.Done(); s.t3.Mul(m1.a, m2.b) }()
		go func() { defer wg.Done(); s.t4.Mul(m1.b, m2.d) }()
		go func() { defer wg.Done(); s.t5.Mul(m1.c, m2.a) }()
		go func() { defer wg.Done(); s.t6.Mul(m1.d, m2.c) }()
		go func() { defer wg.Done(); s.t7.Mul(m1.c, m2.b) }()
		go func() { defer wg.Done(); s.t8.Mul(m1.d, m2.d) }()
		wg.Wait()
	} else {
		s.t1.Mul(m1.a, m2.a)
		s.t2.Mul(m1.b, m2.c)
		s.t3.Mul(m1.a, m2.b)
		s.t4.Mul(m1.b, m2.d)
		s.t5.Mul(m1.c, m2.a)
		s.t6.Mul(m1.d, m2.c)
		s.t7.Mul(m1.c, m2.b)
		s.t8.Mul(m1.d, m2.d)
	}
	dest.a.Add(s.t1, s.t2)
	dest.b.Add(s.t3, s.t4)
	dest.c.Add(s.t5, s.t6)
	dest.d.Add(s.t7, s.t8)
}

func (me *MatrixExponentiation) Calculate(ctx context.Context, progressChan chan<- float64, n uint64) (*big.Int, error) {
	if n == 0 {
		reportProgress(progressChan, 1.0)
		return big.NewInt(0), nil
	}
	if n <= MaxFibUint64 {
		reportProgress(progressChan, 1.0)
		return calculateSmall(n), nil
	}

	s := getMatrixState()
	defer putMatrixState(s)

	k := n - 1
	numBits := bits.Len64(k)
	invNumBits := 1.0 / float64(numBits)
	useParallel := runtime.NumCPU() > 1
	tempMatrix := &matrix{new(big.Int), new(big.Int), new(big.Int), new(big.Int)}

	for i := 0; i < numBits; i++ {
		if ctx.Err() != nil {
			return nil, fmt.Errorf("calculation canceled: %w", ctx.Err())
		}
		if progressChan != nil {
			reportProgress(progressChan, float64(i)*invNumBits)
		}

		if (k>>uint(i))&1 == 1 {
			multiplyMatrices(tempMatrix, s.res, s.p, s, useParallel)
			s.res.a.Set(tempMatrix.a)
			s.res.b.Set(tempMatrix.b)
			s.res.c.Set(tempMatrix.c)
			s.res.d.Set(tempMatrix.d)
		}
		multiplyMatrices(tempMatrix, s.p, s.p, s, useParallel)
		s.p.a.Set(tempMatrix.a)
		s.p.b.Set(tempMatrix.b)
		s.p.c.Set(tempMatrix.c)
		s.p.d.Set(tempMatrix.d)
	}

	reportProgress(progressChan, 1.0)
	return new(big.Int).Set(s.res.a), nil
}

// ----------------------------------------------------------------------------
// SECTION 2: LOGIQUE DE L'APPLICATION (Interface en ligne de commande)
// ----------------------------------------------------------------------------

const (
	ExitSuccess       = 0
	ExitErrorGeneric  = 1
	ExitErrorTimeout  = 2
	ExitErrorCanceled = 130
	ExitErrorMismatch = 3
)

type AppConfig struct {
	N       uint64
	Verbose bool
	Timeout time.Duration
	Algo    string
}

type ProgressUpdate struct {
	CalculatorIndex int
	Value           float64
}

func main() {
	nFlag := flag.Uint64("n", 10000000, "L'indice 'n' de la séquence de Fibonacci à calculer.")
	verboseFlag := flag.Bool("v", false, "Affiche le résultat complet.")
	timeoutFlag := flag.Duration("timeout", 5*time.Minute, "Délai maximum (ex: 30s, 1m).")
	algoFlag := flag.String("algo", "all", "Algorithme : 'fast', 'matrix', ou 'all' (par défaut) pour comparer.")
	flag.Parse()

	config := AppConfig{
		N:       *nFlag,
		Verbose: *verboseFlag,
		Timeout: *timeoutFlag,
		Algo:    *algoFlag,
	}

	exitCode := run(context.Background(), config, os.Stdout)
	os.Exit(exitCode)
}

func run(ctx context.Context, config AppConfig, out io.Writer) int {
	ctx, cancelTimeout := context.WithTimeout(ctx, config.Timeout)
	defer cancelTimeout()
	ctx, stopSignals := signal.NotifyContext(ctx, syscall.SIGINT, syscall.SIGTERM)
	defer stopSignals()

	fmt.Fprintf(out, "Calcul de F(%d)...\n", config.N)
	fmt.Fprintf(out, "Nombre de cœurs CPU disponibles : %d\n", runtime.NumCPU())
	fmt.Fprintf(out, "Timeout défini à %s.\n", config.Timeout)

	algo := strings.ToLower(config.Algo)
	if algo == "all" {
		return runAllAndCompare(ctx, config, out)
	}
	return runSingle(ctx, config, algo, out)
}

func runSingle(ctx context.Context, config AppConfig, algoName string, out io.Writer) int {
	var calculator Calculator
	switch algoName {
	case "fast":
		calculator = &OptimizedFastDoubling{}
	case "matrix":
		calculator = &MatrixExponentiation{}
	default:
		fmt.Fprintf(out, "Erreur : algorithme '%s' inconnu. Choix possibles : 'fast', 'matrix', 'all'.\n", config.Algo)
		return ExitErrorGeneric
	}

	fmt.Fprintf(out, "Algorithme : %s\n", calculator.Name())

	progressChan := make(chan float64, 100)
	var wg sync.WaitGroup
	wg.Add(1)
	go displayProgress(&wg, progressChan, out)

	startTime := time.Now()
	result, err := calculator.Calculate(ctx, progressChan, config.N)
	close(progressChan)
	wg.Wait()
	duration := time.Since(startTime)

	fmt.Fprintln(out, "\n--- Résultats ---")
	if err != nil {
		return handleCalculationError(err, duration, config.Timeout, out)
	}

	displayResult(result, config.N, duration, config.Verbose, out)
	return ExitSuccess
}

func runAllAndCompare(ctx context.Context, config AppConfig, out io.Writer) int {
	calculators := []Calculator{
		&OptimizedFastDoubling{},
		&MatrixExponentiation{},
	}

	type CalculationResult struct {
		Name     string
		Result   *big.Int
		Duration time.Duration
		Err      error
	}

	results := make([]CalculationResult, len(calculators))
	var calcWg, displayWg sync.WaitGroup

	fmt.Fprintf(out, "Mode comparaison : Lancement de %d algorithmes en parallèle...\n", len(calculators))

	globalProgressChan := make(chan ProgressUpdate, len(calculators)*10)
	displayWg.Add(1)
	go displayGlobalProgress(&displayWg, globalProgressChan, len(calculators), out)

	for i, calc := range calculators {
		calcWg.Add(1)
		go func(idx int, calculator Calculator) {
			defer calcWg.Done()

			proxyChan := make(chan float64, 100)
			go func() {
				for p := range proxyChan {
					globalProgressChan <- ProgressUpdate{CalculatorIndex: idx, Value: p}
				}
			}()

			startTime := time.Now()
			res, err := calculator.Calculate(ctx, proxyChan, config.N)
			close(proxyChan)
			duration := time.Since(startTime)

			results[idx] = CalculationResult{
				Name:     calculator.Name(),
				Result:   res,
				Duration: duration,
				Err:      err,
			}
		}(i, calc)
	}

	calcWg.Wait()
	close(globalProgressChan)
	displayWg.Wait()

	fmt.Fprintln(out, "\n--- Résultats de la comparaison ---")

	var firstResult *big.Int
	var firstError error

	for _, res := range results {
		status := "Succès"
		if res.Err != nil {
			status = fmt.Sprintf("Échec (%v)", res.Err)
			if firstError == nil {
				firstError = res.Err
			}
		} else {
			if firstResult == nil {
				firstResult = res.Result
			}
		}
		fmt.Fprintf(out, "  - %-55s -> Durée: %-15s | Statut: %s\n", res.Name, res.Duration.String(), status)
	}

	if firstError != nil {
		fmt.Fprintln(out, "\nUn ou plusieurs calculs ont échoué.")
		return handleCalculationError(firstError, 0, config.Timeout, out)
	}

	for i := 1; i < len(results); i++ {
		if results[i].Result.Cmp(firstResult) != 0 {
			fmt.Fprintln(out, "\nStatut : Échec critique. Les algorithmes ont produit des résultats différents !")
			return ExitErrorMismatch
		}
	}
	fmt.Fprintln(out, "\nValidation : Tous les résultats sont identiques.")

	displayResult(firstResult, config.N, 0, config.Verbose, out)
	return ExitSuccess
}

func handleCalculationError(err error, duration time.Duration, timeout time.Duration, out io.Writer) int {
	msg := ""
	if duration > 0 {
		msg = fmt.Sprintf(" après %s", duration)
	}
	if errors.Is(err, context.DeadlineExceeded) {
		fmt.Fprintf(out, "Statut : Échec. Le calcul a dépassé le délai imparti (%s)%s.\n", timeout, msg)
		return ExitErrorTimeout
	} else if errors.Is(err, context.Canceled) {
		fmt.Fprintf(out, "Statut : Annulé (signal reçu ou Ctrl+C)%s.\n", msg)
		return ExitErrorCanceled
	} else {
		fmt.Fprintf(out, "Statut : Échec. Erreur interne : %v\n", err)
		return ExitErrorGeneric
	}
}

func displayResult(result *big.Int, n uint64, duration time.Duration, verbose bool, out io.Writer) {
	fmt.Fprintln(out, "\n--- Données du résultat ---")
	if duration > 0 {
		fmt.Fprintf(out, "Durée d'exécution : %s\n", duration)
	}
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
		fmt.Fprintf(out, "\rProgression : %6.2f%% [%-30s]", progress*100, progressBar(progress, 30))
	}

	for {
		select {
		case p, ok := <-progressChan:
			if !ok {
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

func displayGlobalProgress(wg *sync.WaitGroup, progressChan <-chan ProgressUpdate, numCalculators int, out io.Writer) {
	defer wg.Done()
	progresses := make([]float64, numCalculators)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	printBar := func() {
		var totalProgress float64
		for _, p := range progresses {
			totalProgress += p
		}
		avgProgress := totalProgress / float64(numCalculators)
		fmt.Fprintf(out, "\rProgression Globale : %6.2f%% [%-30s]", avgProgress*100, progressBar(avgProgress, 30))
	}

	for {
		select {
		case update, ok := <-progressChan:
			if !ok {
				for i := range progresses {
					progresses[i] = 1.0
				}
				printBar()
				fmt.Fprintln(out)
				return
			}
			if update.CalculatorIndex < len(progresses) {
				progresses[update.CalculatorIndex] = update.Value
			}
		case <-ticker.C:
			printBar()
		}
	}
}

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
