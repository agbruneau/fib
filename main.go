package main

import (
	"context"
	"flag"
	"fmt"
	"math/big"
	"math/bits"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"
)

// ----------------------------------------------------------------------------
// Fibonacci Implementation (Simulating the 'fibonacci' package structure)
// This section implements the core logic as defined by the SRS and RTM.
// ----------------------------------------------------------------------------

// --- Interface Definition ---

// Calculator interface defines the contract for Fibonacci algorithms (RTM: REQ-15 Extensibility).
type Calculator interface {
	Calculate(ctx context.Context, progressChan chan<- float64, n uint64, pool *sync.Pool) (*big.Int, error)
	Name() string
}

// --- Memory Management Utilities ---

// calculationTemps centralizes intermediate variables used during calculation (RTM: REQ-06).
type calculationTemps struct {
	// t1, t2, t3 are temporary big.Int used within the loop iterations to avoid allocations.
	t1, t2, t3 *big.Int
}

// getTemps retrieves a set of initialized temporary variables from the pool (RTM: REQ-05).
func getTemps(pool *sync.Pool) *calculationTemps {
	// RTM: REQ-04 (Support big.Int)
	return &calculationTemps{
		t1: pool.Get().(*big.Int),
		t2: pool.Get().(*big.Int),
		t3: pool.Get().(*big.Int),
	}
}

// putTemps returns the temporary variables back to the pool for reuse (RTM: REQ-05).
func putTemps(pool *sync.Pool, temps *calculationTemps) {
	// Resetting (SetInt64(0)) ensures the objects are clean for the next usage,
	// while retaining their internal memory allocation.
	pool.Put(temps.t1.SetInt64(0))
	pool.Put(temps.t2.SetInt64(0))
	pool.Put(temps.t3.SetInt64(0))
}

// reportProgress sends the progress value to the channel non-blockingly (RTM: REQ-09, REQ-11).
func reportProgress(progressChan chan<- float64, progress float64) {
	if progressChan == nil {
		return
	}
	// Use select with default to ensure the calculation is never blocked by the UI/reporting.
	select {
	case progressChan <- progress:
	default:
		// Drop the update if the channel buffer is full.
	}
}

// --- Fast Doubling Implementation ---

// FastDoubling implements the Calculator interface.
type FastDoubling struct{}

func (fd *FastDoubling) Name() string {
	return "FastDoubling"
}

// Calculate computes the nth Fibonacci number using the Fast Doubling algorithm.
// (RTM: REQ-01 Algorithm, REQ-02 O(log n) complexity).
func (fd *FastDoubling) Calculate(ctx context.Context, progressChan chan<- float64, n uint64, pool *sync.Pool) (*big.Int, error) {

	// RTM: REQ-03 - Handle base cases explicitly.
	if n == 0 {
		reportProgress(progressChan, 1.0)
		return big.NewInt(0), nil
	}
	if n == 1 {
		reportProgress(progressChan, 1.0)
		return big.NewInt(1), nil
	}

	// Setup memory management.
	temps := getTemps(pool)
	defer putTemps(pool, temps)

	// Initialize the state (k=0): F(0)=0, F(1)=1.
	// We use explicit naming (RTM: REQ-13).
	// f_k represents F(k)
	// f_k1 represents F(k+1)
	f_k := big.NewInt(0)
	f_k1 := big.NewInt(1)

	// Determine the number of bits in n.
	numBits := bits.Len64(n)

	// Iterate through the bits of n from most significant (MSB) to least significant (LSB).
	// RTM: REQ-14 - Comment explaining the binary approach.
	// We iterate from i = numBits - 1 down to 0.
	for i := numBits - 1; i >= 0; i-- {

		// RTM: REQ-07 - Check for cancellation signal from the context periodically.
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
			// Proceed with calculation.
		}

		// RTM: REQ-10 - Calculate progression based on bits traversed.
		if i < numBits-1 {
			// Progress is the ratio of bits processed so far.
			progress := float64(numBits-1-i) / float64(numBits)
			reportProgress(progressChan, progress)
		}

		// --- Doubling Step (k -> 2k) ---
		// Calculate F(2k) and F(2k+1) based on F(k) and F(k+1).

		// RTM: REQ-14 - Comment explaining the formulas.

		// 1. Calculate F(2k) = F(k) * [2*F(k+1) - F(k)]

		// t1 = 2 * F(k+1)
		temps.t1.Lsh(f_k1, 1) // Efficient multiplication by 2 (Left shift).
		// t2 = [2*F(k+1) - F(k)]
		temps.t2.Sub(temps.t1, f_k)
		// F(2k) (stored temporarily in t3) = F(k) * t2
		temps.t3.Mul(f_k, temps.t2)

		// 2. Calculate F(2k+1) = F(k+1)^2 + F(k)^2

		// t1 = F(k+1)^2
		temps.t1.Mul(f_k1, f_k1)
		// t2 = F(k)^2
		temps.t2.Mul(f_k, f_k)
		// F(2k+1) (new f_k1) = t1 + t2
		f_k1.Add(temps.t1, temps.t2)

		// Update F(2k) (moved from t3 to f_k)
		f_k.Set(temps.t3)

		// --- Addition Step (k -> k+1 if necessary) ---
		// If the current bit of n at position i is 1, we advance the state by one.
		if (n>>uint(i))&1 == 1 {
			// (F(k'), F(k'+1)) becomes (F(k'+1), F(k'+2))
			// where k' is the previously calculated 2k.
			// F(k'+2) = F(k'+1) + F(k')

			// t1 (new F(k)) = current F(k+1)
			temps.t1.Set(f_k1)
			// new F(k+1) = current F(k+1) + current F(k)
			f_k1.Add(f_k1, f_k)
			// f_k = t1
			f_k.Set(temps.t1)
		}
	}

	reportProgress(progressChan, 1.0)
	// After iterating through all bits, f_k holds F(n).
	return f_k, nil
}

// ----------------------------------------------------------------------------
// Main Function (Entry Point and CLI Application)
// This section demonstrates how to use the Fibonacci implementation.
// ----------------------------------------------------------------------------

// Initialize the global pool for big.Int reuse (RTM: REQ-05).
// This ensures thread-safe access to the pool (RTM: REQ-08).
var bigIntPool = sync.Pool{
	New: func() interface{} {
		return new(big.Int)
	},
}

func main() {
	// 1. Configuration des arguments CLI
	nFlag := flag.Uint64("n", 1000, "L'indice 'n' de la séquence de Fibonacci à calculer.")
	verboseFlag := flag.Bool("v", false, "Affiche le résultat complet (peut être très long).")
	timeoutFlag := flag.Duration("timeout", 5*time.Minute, "Délai maximum pour le calcul (ex: 30s, 1m).")
	flag.Parse()

	n := *nFlag
	fmt.Printf("Calcul de F(%d) avec l'algorithme Fast Doubling...\n", n)
	fmt.Printf("Timeout défini à %s.\n", *timeoutFlag)

	// 2. Sélection de la Stratégie
	var calculator Calculator = &FastDoubling{}

	// 3. Configuration du Contexte et de l'Annulation (RTM: REQ-07)
	// 3a. Contexte avec Timeout
	ctx, cancelTimeout := context.WithTimeout(context.Background(), *timeoutFlag)
	defer cancelTimeout()

	// 3b. Contexte avec gestion des signaux (Ctrl+C)
	// Ce contexte hérite du timeout précédent et ajoute la gestion des signaux.
	ctx, stopSignals := signal.NotifyContext(ctx, syscall.SIGINT, syscall.SIGTERM)
	defer stopSignals()

	// 4. Configuration du Reporting de Progression (RTM: REQ-09)
	// Canal bufferisé pour gérer les mises à jour rapides sans bloquer le calcul.
	progressChan := make(chan float64, 100)
	var wg sync.WaitGroup

	// Lancement d'une goroutine dédiée à l'affichage de la progression.
	wg.Add(1)
	go displayProgress(&wg, progressChan)

	// 5. Exécution du Calcul
	startTime := time.Now()
	result, err := calculator.Calculate(ctx, progressChan, n, &bigIntPool)

	// Fermeture du canal de progression une fois le calcul terminé (ou annulé).
	close(progressChan)
	// Attente que l'affichage de la progression soit terminé.
	wg.Wait()

	duration := time.Since(startTime)

	// 6. Gestion des Résultats et Erreurs
	fmt.Println("\n--- Résultats ---")
	if err != nil {
		// Vérification de la cause de l'erreur via le contexte.
		switch ctx.Err() {
		case context.DeadlineExceeded:
			fmt.Printf("Statut : Échec. Le calcul a dépassé le délai imparti (%s) après %s.\n", *timeoutFlag, duration)
			os.Exit(2)
		case context.Canceled:
			fmt.Printf("Statut : Annulé par l'utilisateur (Ctrl+C) après %s.\n", duration)
			os.Exit(130) // Code d'erreur standard pour Ctrl+C
		default:
			fmt.Printf("Statut : Échec. Erreur interne : %v\n", err)
			os.Exit(1)
		}
	}

	// 7. Affichage du Résultat
	fmt.Printf("Statut : Succès\n")
	fmt.Printf("Durée d'exécution : %s\n", duration)
	fmt.Printf("Taille du résultat : %d bits.\n", result.BitLen())

	resultStr := result.String()
	fmt.Printf("Nombre de chiffres décimaux : %d\n", len(resultStr))

	if *verboseFlag {
		fmt.Printf("F(%d) = %s\n", n, resultStr)
	} else if len(resultStr) > 50 {
		// Affichage tronqué pour les grands nombres.
		fmt.Printf("F(%d) (tronqué) = %s...%s\n", n, resultStr[:20], resultStr[len(resultStr)-20:])
	} else {
		fmt.Printf("F(%d) = %s\n", n, resultStr)
	}
}

// displayProgress lit le canal de progression et met à jour l'affichage périodiquement.
func displayProgress(wg *sync.WaitGroup, progressChan <-chan float64) {
	defer wg.Done()
	// Utilisation d'un ticker pour limiter le taux de rafraîchissement de l'affichage.
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	lastProgress := 0.0
	done := false

	for !done {
		select {
		case p, ok := <-progressChan:
			if !ok {
				// Le canal est fermé, le calcul est terminé.
				lastProgress = 1.0
				done = true
			} else {
				lastProgress = p
			}
		case <-ticker.C:
			// Mise à jour de l'affichage à intervalle régulier.
			fmt.Printf("\rProgression : %6.2f%% [%-30s]", lastProgress*100, progressBar(lastProgress, 30))
		}
	}
	// Affichage final à 100%.
	fmt.Printf("\rProgression : 100.00%% [%-30s]\n", progressBar(1.0, 30))
}

// Fonction utilitaire pour la barre de progression CLI.
func progressBar(progress float64, length int) string {
	count := int(progress * float64(length))
	if count > length {
		count = length
	}
	if count < 0 {
		count = 0
	}
	bar := make([]rune, length)
	for i := 0; i < length; i++ {
		if i < count {
			bar[i] = '■'
		} else {
			bar[i] = ' '
		}
	}
	return string(bar)
}
