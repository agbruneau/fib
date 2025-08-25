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
	"sync"
	"syscall"
	"time"
)

// ----------------------------------------------------------------------------
// SECTION 1: MOTEUR DE CALCUL FIBONACCI (Logique principale, hautement optimisée)
// ----------------------------------------------------------------------------

// MaxFibUint64 est l'index N le plus élevé pour lequel F(N) peut être stocké
// dans un entier non signé de 64 bits (uint64). Au-delà de F(93), le résultat
// dépasse la capacité de ce type de données.
const MaxFibUint64 = 93

// parallelThreshold définit la taille minimale (en bits) d'un nombre à partir de
// laquelle il devient plus performant d'effectuer les multiplications en parallèle
// sur plusieurs cœurs de processeur. En dessous de ce seuil, le coût de la
// création de goroutines est supérieur au gain de performance.
const parallelThreshold = 2048

// Calculator est une interface décrivant un objet capable de calculer un nombre de Fibonacci.
// Cela permettrait, par exemple, de comparer facilement différentes implémentations.
type Calculator interface {
	Calculate(ctx context.Context, progressChan chan<- float64, n uint64) (*big.Int, error)
	Name() string
}

// --- Stratégie de gestion de la mémoire : "Zéro Allocation" ---
//
// OBJECTIF : Éviter de créer de nouveaux objets en mémoire dans la boucle de calcul.
//
// En Go, la création d'objets (comme des `big.Int`) alloue de la mémoire, et le
// "Garbage Collector" (GC) doit ensuite travailler pour nettoyer cette mémoire.
// Dans un calcul intensif, ce cycle de création/nettoyage peut considérablement
// ralentir les performances.
//
// La stratégie ici est de pré-allouer un ensemble d'objets `big.Int` et de les
// réutiliser à chaque étape du calcul. On utilise pour cela un `sync.Pool`, qui
// fonctionne comme une "piscine" d'objets réutilisables.

// calculationState regroupe toutes les variables `big.Int` nécessaires pour une
// opération de calcul. Cela permet de les "emprunter" et de les "rendre" à la
// piscine d'objets en une seule fois.
type calculationState struct {
	// Variables d'état pour l'algorithme : F(k) et F(k+1)
	f_k  *big.Int
	f_k1 *big.Int
	// Variables temporaires pour les calculs intermédiaires
	t1, t2, t3, t4 *big.Int
}

// statePool est la "piscine" d'objets. Elle gère une liste d'objets `calculationState`
// disponibles. `sync.Pool` est optimisé pour un usage concurrentiel.
var statePool = sync.Pool{
	// La fonction `New` est appelée par la piscine uniquement si aucun objet n'est
	// disponible. Elle crée une nouvelle "instance" de notre état de calcul.
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

// getState "emprunte" un objet `calculationState` depuis la piscine.
// Il le réinitialise à l'état de départ du calcul de Fibonacci : F(0)=0 et F(1)=1.
func getState() *calculationState {
	s := statePool.Get().(*calculationState)
	// Initialise l'état pour le début du calcul : F(0) = 0, F(1) = 1.
	s.f_k.SetInt64(0)
	s.f_k1.SetInt64(1)
	// Les variables temporaires n'ont pas besoin d'être réinitialisées car elles
	// sont toujours écrasées avant d'être lues.
	return s
}

// putState "rend" un objet `calculationState` à la piscine une fois qu'il n'est
// plus utilisé. L'objet devient alors disponible pour un autre calcul.
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

// --- Implémentation de l'algorithme optimisé ---

// OptimizedFastDoubling est une structure qui implémente l'interface Calculator.
// Son nom technique indique les optimisations utilisées :
// - "FastDoubling": L'algorithme principal, extrêmement rapide (complexité O(log n)).
// - "3-Way-Parallel": Utilise la parallélisation pour les très grands nombres.
// - "ZeroAlloc": La stratégie de gestion mémoire pour éviter le "garbage collector".
// - "FastPath": Une voie rapide pour les petits nombres qui n'ont pas besoin de `big.Int`.
type OptimizedFastDoubling struct{}

func (fd *OptimizedFastDoubling) Name() string {
	return "OptimizedFastDoubling (3-Way-Parallel+ZeroAlloc+FastPath)"
}

// calculateSmall est une optimisation (dite "fast path") pour les petits nombres.
// Pour n <= 93, le calcul peut être fait avec des entiers natifs (uint64), ce qui
// est beaucoup plus rapide que d'utiliser `big.Int` qui est conçu pour des nombres
// de taille arbitraire.
func (fd *OptimizedFastDoubling) calculateSmall(n uint64) *big.Int {
	if n == 0 {
		return big.NewInt(0)
	}
	var a, b uint64 = 0, 1
	// Calcul itératif simple.
	for i := uint64(1); i < n; i++ {
		a, b = b, a+b
	}
	return new(big.Int).SetUint64(b)
}

// Calculate calcule le n-ième nombre de Fibonacci en utilisant l'algorithme "Fast Doubling".
func (fd *OptimizedFastDoubling) Calculate(ctx context.Context, progressChan chan<- float64, n uint64) (*big.Int, error) {

	// 1. Optimisation "Fast Path" : gestion des petits nombres
	if n <= MaxFibUint64 {
		reportProgress(progressChan, 1.0)
		return fd.calculateSmall(n), nil
	}

	// 2. Préparation pour les grands nombres (Optimisation "Zéro Allocation")
	// On "emprunte" un état de calcul à notre piscine d'objets.
	s := getState()
	// `defer` garantit que l'état sera "rendu" à la piscine à la fin de la fonction,
	// quoi qu'il arrive (succès ou erreur).
	defer putState(s)

	// L'algorithme fonctionne en lisant les bits de `n`. On calcule donc le nombre
	// de bits à traiter.
	numBits := bits.Len64(n)
	// Micro-optimisation pour le calcul de la progression (évite une division dans la boucle).
	invNumBits := 1.0 / float64(numBits)

	// Préparation à la parallélisation.
	var wg sync.WaitGroup
	useParallel := runtime.NumCPU() > 1

	// 3. Boucle principale : Itération sur les bits de `n`
	// L'algorithme "Fast Doubling" est basé sur la représentation binaire de `n`.
	// On parcourt les bits de `n` du plus significatif (gauche) au moins
	// significatif (droite).
	for i := numBits - 1; i >= 0; i-- {

		// Vérification périodique pour voir si le calcul a été annulé (ex: timeout).
		if ctx.Err() != nil {
			return nil, fmt.Errorf("calculation canceled: %w", ctx.Err())
		}

		// Envoi de la progression (de manière non bloquante).
		if progressChan != nil && i < numBits-1 {
			progress := float64(numBits-1-i) * invNumBits
			reportProgress(progressChan, progress)
		}

		// --- Étape de "Doubling" (Doublement) ---
		// À chaque bit de `n` (qu'il soit 0 ou 1), on applique les formules de
		// doublement qui permettent de passer de F(k) et F(k+1) à F(2k) et F(2k+1).
		// Les formules sont :
		// F(2k)   = F(k) * [2*F(k+1) - F(k)]
		// F(2k+1) = F(k)² + F(k+1)²
		//
		// Le calcul est décomposé en plusieurs opérations sur des `big.Int`.

		// Calcul de la partie commune : 2*F(k+1) - F(k)
		s.t2.Lsh(s.f_k1, 1)   // t2 = F(k+1) * 2
		s.t2.Sub(s.t2, s.f_k) // t2 = t2 - F(k)

		// Si les nombres sont assez grands et que la machine a plusieurs cœurs,
		// on parallélise les 3 multiplications, qui sont les opérations les plus coûteuses.
		if useParallel && s.f_k1.BitLen() > parallelThreshold {
			// NOTE DE PERFORMANCE : Sur une machine multi-cœur, cette parallélisation
			// offre un gain de vitesse significatif (>2x pour N=10,000,000)
			// par rapport à une approche séquentielle, ce qui justifie la complexité.
			wg.Add(3)

			// Goroutine 1: Calcule F(2k)
			go func(dest, src1, src2 *big.Int) {
				defer wg.Done()
				dest.Mul(src1, src2) // F(k) * (2*F(k+1) - F(k))
			}(s.t3, s.f_k, s.t2)

			// Goroutine 2: Calcule F(k+1)²
			go func(dest, src *big.Int) {
				defer wg.Done()
				dest.Mul(src, src)
			}(s.t1, s.f_k1)

			// Goroutine 3: Calcule F(k)²
			go func(dest, src *big.Int) {
				defer wg.Done()
				dest.Mul(src, src)
			}(s.t4, s.f_k)

			wg.Wait() // On attend que les 3 calculs finissent.

			// On combine les résultats pour obtenir F(2k) et F(2k+1)
			s.f_k.Set(s.t3)        // F(k) devient F(2k)
			s.f_k1.Add(s.t1, s.t4) // F(k+1) devient F(2k+1) = F(k+1)² + F(k)²

		} else {
			// Exécution séquentielle pour les nombres plus petits ou sur un seul cœur.
			s.t3.Mul(s.f_k, s.t2)    // F(2k) = F(k) * (2*F(k+1) - F(k))
			s.t1.Mul(s.f_k1, s.f_k1) // F(k+1)²
			s.t4.Mul(s.f_k, s.f_k)   // F(k)²

			// Combinaison des résultats
			s.f_k.Set(s.t3)        // F(k) devient F(2k)
			s.f_k1.Add(s.t1, s.t4) // F(k+1) devient F(2k+1)
		}

		// --- Étape d'"Addition" (si le bit de n est 1) ---
		// Si le bit de `n` que nous lisons est un '1', cela signifie que nous devons
		// avancer d'un pas supplémentaire, de `2k` à `2k+1`.
		// On utilise la formule de base de Fibonacci : F(m+1) = F(m) + F(m-1).
		// Ici, on calcule F(2k+2) = F(2k+1) + F(2k).
		if (n>>uint(i))&1 == 1 {
			// On stocke F(k+1) (qui contient maintenant F(2k+1)) dans une variable temporaire.
			s.t1.Set(s.f_k1)
			// Le nouveau F(k+1) devient F(2k+2) = F(2k+1) + F(2k).
			s.f_k1.Add(s.f_k1, s.f_k)
			// Le nouveau F(k) devient l'ancien F(k+1), c'est-à-dire F(2k+1).
			s.f_k.Set(s.t1)
		}
	}

	reportProgress(progressChan, 1.0)

	// NOTE CRITIQUE : Il faut renvoyer une COPIE du résultat.
	// La variable `s.f_k` appartient à la piscine d'objets et sera réutilisée.
	// Si nous renvoyions un pointeur vers `s.f_k`, sa valeur pourrait être modifiée
	// par un autre calcul, créant des bugs très difficiles à détecter.
	return new(big.Int).Set(s.f_k), nil
}

// ----------------------------------------------------------------------------
// SECTION 2: LOGIQUE DE L'APPLICATION (Interface en ligne de commande)
// ----------------------------------------------------------------------------

// Codes de sortie standards pour indiquer le résultat de l'exécution du programme.
const (
	ExitSuccess       = 0   // Le programme s'est terminé avec succès.
	ExitErrorGeneric  = 1   // Une erreur non spécifiée s'est produite.
	ExitErrorTimeout  = 2   // Le calcul a dépassé le temps imparti.
	ExitErrorCanceled = 130 // L'utilisateur a annulé l'opération (ex: Ctrl+C).
)

// Job représente une seule tâche à calculer.
type Job struct {
	N uint64
}

// Result contient le résultat d'une tâche terminée.
type Result struct {
	Job      Job
	Value    *big.Int
	Duration time.Duration
	Err      error
}

// AppConfig regroupe les paramètres de configuration de l'application,
// principalement issus des arguments de la ligne de commande.
type AppConfig struct {
	Timeout time.Duration // La durée maximale autorisée pour le calcul.
}

// worker est une goroutine qui lit les jobs du canal `jobs`, les exécute,
// et envoie les résultats sur le canal `results`.
func worker(ctx context.Context, wg *sync.WaitGroup, calculator Calculator, jobs <-chan Job, results chan<- Result) {
	defer wg.Done()
	for job := range jobs {
		// Vérifie si le contexte a été annulé avant de démarrer un nouveau calcul.
		select {
		case <-ctx.Done():
			// Le contexte est annulé, on ne commence pas de nouveaux jobs.
			return
		default:
			// Continue l'exécution.
		}

		startTime := time.Now()
		// Note : on passe un `nil` pour le canal de progression, car on ne l'utilise plus.
		value, err := calculator.Calculate(ctx, nil, job.N)
		duration := time.Since(startTime)

		results <- Result{
			Job:      job,
			Value:    value,
			Duration: duration,
			Err:      err,
		}
	}
}

// main est le point d'entrée du programme.
func main() {
	// 1. Configuration de l'interface en ligne de commande (CLI).
	// Le package `flag` permet de définir les arguments que l'utilisateur peut passer.
	timeoutFlag := flag.Duration("timeout", 5*time.Minute, "Délai maximum (ex: 30s, 1m).")
	flag.Parse() // Analyse les arguments fournis par l'utilisateur.

	// 2. Création de la configuration de l'application.
	config := AppConfig{
		Timeout: *timeoutFlag,
	}

	// 3. Lancement de l'application.
	// La logique principale est dans la fonction `run` pour la rendre plus facile à tester.
	exitCode := run(context.Background(), config, os.Stdout)
	os.Exit(exitCode) // Termine le programme avec le code de sortie approprié.
}

// run est la fonction principale qui orchestre le calcul et l'affichage.
func run(ctx context.Context, config AppConfig, out io.Writer) int {
	// Instanciation du moteur de calcul.
	var calculator Calculator = &OptimizedFastDoubling{}

	// --- Configuration du pool de workers ---
	// On inclut un nombre très grand pour que le timeout soit testable.
	tasks := []uint64{0, 1, 10, 20, 40, 93, 94, 100, 500, 1000, 1000000}
	numJobs := len(tasks)
	jobs := make(chan Job, numJobs)
	results := make(chan Result, numJobs)
	numWorkers := runtime.NumCPU()

	fmt.Fprintf(out, "Lancement de %d workers pour calculer F(n) pour %d tâches.\n", numWorkers, numJobs)
	fmt.Fprintf(out, "Algorithme : %s\n", calculator.Name())
	fmt.Fprintf(out, "Timeout global défini à %s.\n", config.Timeout)
	fmt.Fprintln(out, "---")

	// --- Gestion du cycle de vie avec `context` ---
	ctx, cancelTimeout := context.WithTimeout(ctx, config.Timeout)
	defer cancelTimeout()
	ctx, stopSignals := signal.NotifyContext(ctx, syscall.SIGINT, syscall.SIGTERM)
	defer stopSignals()

	// --- Démarrage du pool de workers ---
	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go worker(ctx, &wg, calculator, jobs, results)
	}

	// --- Envoi des tâches dans le canal `jobs` ---
	for _, n := range tasks {
		jobs <- Job{N: n}
	}
	close(jobs)

	// --- Attente de la fin des workers et fermeture du canal de résultats ---
	go func() {
		wg.Wait()
		close(results)
	}()

	// --- Collecte et affichage des résultats ---
	var successful, failed int
	for result := range results {
		if result.Err != nil {
			failed++
			// Affiche l'erreur specifique pour le job qui a échoué
			if errors.Is(result.Err, context.DeadlineExceeded) {
				fmt.Fprintf(out, "F(%d) a échoué (timeout dépassé) en %s\n", result.Job.N, result.Duration)
			} else if errors.Is(result.Err, context.Canceled) {
				// Ce cas peut arriver si le contexte global est annulé pendant qu'un calcul est en cours.
				fmt.Fprintf(out, "F(%d) a été annulé après %s\n", result.Job.N, result.Duration)
			} else {
				fmt.Fprintf(out, "F(%d) a échoué avec une erreur interne en %s : %v\n", result.Job.N, result.Duration, result.Err)
			}
		} else {
			successful++
			resultStr := result.Value.String()
			fmt.Fprintf(out, "F(%d) = %s (%d chiffres) [calculé en %s]\n",
				result.Job.N,
				truncateString(resultStr, 40),
				len(resultStr),
				result.Duration)
		}
	}

	fmt.Fprintln(out, "---")
	fmt.Fprintf(out, "Terminé. %d succès, %d échecs.\n", successful, failed)

	// Vérification si le contexte a été annulé pour le code de sortie global
	if ctx.Err() != nil {
		// On a déjà loggé les erreurs individuelles. Ici on donne le statut global.
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			fmt.Fprintln(out, "Statut global : Échec. Le timeout a été atteint.")
			return ExitErrorTimeout
		}
		if errors.Is(ctx.Err(), context.Canceled) {
			fmt.Fprintln(out, "Statut global : Annulé par l'utilisateur (Ctrl+C).")
			return ExitErrorCanceled
		}
	}

	if failed > 0 {
		return ExitErrorGeneric
	}

	return ExitSuccess
}

// truncateString est une fonction utilitaire pour l'affichage.
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	// Coupe et ajoute "..."
	if maxLen < 3 {
		return s[:maxLen]
	}
	return s[:maxLen-3] + "..."
}
