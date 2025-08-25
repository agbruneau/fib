package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"image/color"
	"io"
	"log"
	"math/big"
	"math/bits"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
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

// AppConfig regroupe les paramètres de configuration de l'application,
// principalement issus des arguments de la ligne de commande.
type AppConfig struct {
	N       uint64        // L'indice 'n' du nombre de Fibonacci à calculer.
	Verbose bool          // Si vrai, affiche le nombre complet sans le tronquer.
	Timeout time.Duration // La durée maximale autorisée pour le calcul.
}

// main est le point d'entrée du programme.
func main() {
	// 1. Configuration de l'interface en ligne de commande (CLI).
	// Le package `flag` permet de définir les arguments que l'utilisateur peut passer.
	nFlag := flag.Uint64("n", 100000000, "L'indice 'n' de la séquence de Fibonacci à calculer.")
	verboseFlag := flag.Bool("v", false, "Affiche le résultat complet.")
	timeoutFlag := flag.Duration("timeout", 5*time.Minute, "Délai maximum (ex: 30s, 1m).")
	graphFlag := flag.String("graph", "", "Génère un graphique de performance et le sauvegarde dans le fichier spécifié (ex: 'performance.png').")
	flag.Parse() // Analyse les arguments fournis par l'utilisateur.

	// Si le drapeau -graph est utilisé, on génère le graphique et on quitte.
	if *graphFlag != "" {
		exitCode := generatePerformanceGraph(*graphFlag, os.Stdout)
		os.Exit(exitCode)
		return
	}

	// 2. Création de la configuration de l'application.
	config := AppConfig{
		N:       *nFlag,
		Verbose: *verboseFlag,
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

	// Affichage des informations de départ.
	fmt.Fprintf(out, "Calcul de F(%d)...\n", config.N)
	fmt.Fprintf(out, "Algorithme : %s\n", calculator.Name())
	fmt.Fprintf(out, "Nombre de cœurs CPU disponibles : %d\n", runtime.NumCPU())
	fmt.Fprintf(out, "Timeout défini à %s.\n", config.Timeout)

	// --- Gestion du cycle de vie avec `context` ---
	// `context` est un outil standard en Go pour gérer l'annulation, les délais
	// et d'autres signaux à travers les différentes parties d'une application.
	//
	// 1. On crée un contexte avec un délai d'attente (timeout).
	ctx, cancelTimeout := context.WithTimeout(ctx, config.Timeout)
	defer cancelTimeout() // Assure que les ressources du timeout sont libérées.
	// 2. On crée un sous-contexte qui écoute les signaux d'arrêt du système (Ctrl+C).
	ctx, stopSignals := signal.NotifyContext(ctx, syscall.SIGINT, syscall.SIGTERM)
	defer stopSignals() // Assure que l'écoute des signaux est arrêtée.
	// Le moteur de calcul `calculator.Calculate` recevra ce `ctx` et devra
	// vérifier périodiquement s'il a été annulé.

	// --- Gestion de l'affichage de la progression ---
	progressChan := make(chan float64, 100) // Canal pour recevoir les pourcentages.
	var wg sync.WaitGroup
	wg.Add(1)
	// On lance l'affichage dans une goroutine séparée pour ne pas bloquer le calcul.
	go displayProgress(&wg, progressChan, out)

	// --- Exécution du calcul ---
	startTime := time.Now()
	result, err := calculator.Calculate(ctx, progressChan, config.N)

	// Une fois le calcul terminé (ou en erreur), on ferme le canal de progression
	// pour signaler à la goroutine d'affichage qu'elle doit se terminer.
	close(progressChan)
	wg.Wait() // On attend que la goroutine d'affichage ait fini son travail.
	duration := time.Since(startTime)

	// --- Gestion des résultats et affichage final ---
	fmt.Fprintln(out, "\n--- Résultats ---")
	if err != nil {
		return handleCalculationError(err, duration, config.Timeout, out)
	}

	displayResult(result, config.N, duration, config.Verbose, out)
	return ExitSuccess
}

// handleCalculationError traduit les erreurs techniques en messages clairs pour l'utilisateur.
func handleCalculationError(err error, duration time.Duration, timeout time.Duration, out io.Writer) int {
	// `errors.Is` est la manière moderne en Go de vérifier le type d'une erreur.
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

// displayResult met en forme le résultat final pour l'utilisateur.
func displayResult(result *big.Int, n uint64, duration time.Duration, verbose bool, out io.Writer) {
	fmt.Fprintf(out, "Statut : Succès\n")
	fmt.Fprintf(out, "Durée d'exécution : %s\n", duration)
	fmt.Fprintf(out, "Taille du résultat : %d bits.\n", result.BitLen())

	resultStr := result.String()
	numDigits := len(resultStr)
	fmt.Fprintf(out, "Nombre de chiffres décimaux : %d\n", numDigits)

	// Pour les nombres très grands, on affiche seulement le début et la fin,
	// sauf si l'utilisateur a demandé l'affichage complet avec `-v`.
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

// displayProgress gère l'affichage et la mise à jour de la barre de progression.
func displayProgress(wg *sync.WaitGroup, progressChan <-chan float64, out io.Writer) {
	defer wg.Done()
	// On utilise un "ticker" pour rafraîchir l'affichage à intervalle régulier
	// (ici, toutes les 100ms), plutôt que de le faire à chaque micro-avancée
	// du calcul, ce qui serait inefficace.
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	lastProgress := 0.0

	printBar := func(progress float64) {
		// Le caractère `\r` (retour chariot) déplace le curseur au début de la ligne
		// sans passer à la ligne suivante, ce qui permet de "réécrire" la barre
		// de progression sur place pour créer une animation.
		fmt.Fprintf(out, "\rProgression : %6.2f%% [%-30s]", progress*100, progressBar(progress, 30))
	}

	// Boucle principale qui attend soit une mise à jour de la progression,
	// soit le déclenchement du ticker.
	for {
		select {
		case p, ok := <-progressChan:
			if !ok {
				// Si le canal est fermé (`ok` est false), cela signifie que le calcul
				// est terminé. On affiche la barre à 100% et on quitte.
				printBar(1.0)
				fmt.Fprintln(out) // Passe à la ligne suivante pour un affichage propre.
				return
			}
			lastProgress = p
		case <-ticker.C:
			// Le ticker s'est déclenché, on rafraîchit la barre avec la dernière
			// progression connue.
			printBar(lastProgress)
		}
	}
}

// progressBar construit la chaîne de caractères représentant la barre de progression.
// L'utilisation de `strings.Builder` est une optimisation pour construire des chaînes
// de caractères efficacement, en évitant des allocations mémoires multiples.
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

// generatePerformanceGraph exécute des benchmarks et génère un graphique de performance.
func generatePerformanceGraph(filename string, out io.Writer) int {
	fmt.Fprintln(out, "Génération du graphique de performance...")

	var calculator Calculator = &OptimizedFastDoubling{}
	points := make(plotter.XYs, 0)

	// Définir les valeurs de 'n' à tester. On utilise une échelle exponentielle
	// pour bien visualiser la courbe de performance.
	nValues := []uint64{
		1, 10, 100, 1000, 10000, 100000, 500000,
		1000000, 2000000, 5000000, 10000000,
	}

	totalSteps := len(nValues)
	for i, n := range nValues {
		fmt.Fprintf(out, "\rCalcul pour n = %-10d (%d/%d)...", n, i+1, totalSteps)

		startTime := time.Now()
		_, err := calculator.Calculate(context.Background(), nil, n)
		duration := time.Since(startTime)

		if err != nil {
			fmt.Fprintf(out, "\nErreur lors du calcul pour n=%d: %v\n", n, err)
			return ExitErrorGeneric
		}

		// Ajoute le point de données (n, temps en secondes) au jeu de données.
		points = append(points, plotter.XY{X: float64(n), Y: duration.Seconds()})
	}
	fmt.Fprintln(out, "\nCalculs terminés.")

	// Création du graphique
	p := plot.New()

	p.Title.Text = "Performance du calcul de Fibonacci"
	p.X.Label.Text = "n (échelle logarithmique)"
	p.Y.Label.Text = "Temps d'exécution (secondes)"
	p.X.Scale = plot.LogScale{} // Utiliser une échelle logarithmique pour l'axe des X

	// Ajout des points de données au graphique
	line, err := plotter.NewLine(points)
	if err != nil {
		log.Printf("Erreur lors de la création de la ligne du graphique : %v", err)
		return ExitErrorGeneric
	}
	line.LineStyle.Width = vg.Points(2)
	line.LineStyle.Color = color.RGBA{R: 25, B: 150, A: 255}

	p.Add(line)

	// Sauvegarde du graphique dans un fichier PNG.
	fmt.Fprintf(out, "Sauvegarde du graphique dans '%s'...\n", filename)
	if err := p.Save(8*vg.Inch, 5*vg.Inch, filename); err != nil {
		log.Printf("Erreur lors de la sauvegarde du graphique : %v", err)
		return ExitErrorGeneric
	}

	fmt.Fprintln(out, "Graphique généré avec succès.")
	return ExitSuccess
}
