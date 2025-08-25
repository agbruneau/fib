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
// SECTION 1: LOGIQUE MÉTIER (Moteur de calcul de Fibonacci - Très optimisé)
// ----------------------------------------------------------------------------

// MaxFibUint64 est l'indice N le plus élevé pour lequel F(N) peut être stocké
// dans un entier non signé de 64 bits (uint64). Au-delà de 93, un `big.Int` est nécessaire.
const MaxFibUint64 = 93

// parallelThreshold définit la taille minimale en bits d'un nombre pour que sa mise
// au carré soit exécutée en parallèle. En dessous de ce seuil, le coût de la
// synchronisation des goroutines dépasse le gain de performance.
const parallelThreshold = 2048

// Calculator définit l'interface pour nos calculateurs de Fibonacci.
// Cela permet de substituer facilement différentes implémentations si nécessaire.
type Calculator interface {
	Calculate(ctx context.Context, progressChan chan<- float64, n uint64) (*big.Int, error)
	Name() string
}

// --- Gestion de la Mémoire (O2: Stratégie Zéro Allocation) ---
//
// L'une des optimisations les plus importantes de ce code.
// Pour des nombres très grands, Go doit allouer de la mémoire sur le "tas" (heap).
// Le "Garbage Collector" (GC) doit alors périodiquement nettoyer cette mémoire,
// ce qui consomme du temps CPU.
//
// La stratégie ici est de ne JAMAIS allouer de mémoire pendant la boucle de calcul principale.
// Nous pré-allouons un "état" de calcul (`calculationState`) via un `sync.Pool`.
// Cet état contient tous les objets `big.Int` dont nous aurons besoin.
// Au lieu de créer de nouveaux objets, nous les réutilisons, réduisant le travail du GC
// à presque zéro et augmentant ainsi considérablement les performances.

// calculationState regroupe TOUTES les variables `big.Int` nécessaires pour un calcul.
// En les gardant dans une seule structure, nous pouvons les mettre en commun (pool) efficacement.
type calculationState struct {
	// Variables d'état principales de l'algorithme "Fast Doubling"
	f_k  *big.Int // Représente F(k) à une étape donnée
	f_k1 *big.Int // Représente F(k+1) à la même étape

	// Variables temporaires, réutilisées pour tous les calculs intermédiaires.
	// t4 a été ajouté spécifiquement pour les opérations parallèles (O3) afin
	// d'éviter les "data races" (accès concurrents à la même mémoire).
	t1, t2, t3, t4 *big.Int
}

// statePool est notre "piscine" d'objets `calculationState`.
// `sync.Pool` est un mécanisme de Go conçu pour la réutilisation d'objets gourmands en mémoire.
var statePool = sync.Pool{
	// La fonction `New` est appelée par le pool uniquement si aucun objet n'est disponible.
	// Elle crée notre état une bonne fois pour toutes.
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

// getState récupère un `calculationState` depuis le pool et l'initialise.
func getState() *calculationState {
	s := statePool.Get().(*calculationState)
	// Initialise l'état pour le début de l'algorithme : F(0)=0 et F(1)=1.
	s.f_k.SetInt64(0)
	s.f_k1.SetInt64(1)
	// Les variables temporaires (t1, t2, ...) n'ont pas besoin d'être réinitialisées
	// car elles sont toujours écrasées avant d'être lues.
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

// --- Implémentation Optimisée ---

// OptimizedFastDoubling est notre implémentation principale, combinant plusieurs
// techniques d'optimisation.
type OptimizedFastDoubling struct{}

// Name retourne le nom de la stratégie de calcul.
func (fd *OptimizedFastDoubling) Name() string {
	return "OptimizedFastDoubling (Parallel+ZeroAlloc+FastPath)"
}

// O1: Chemin Rapide (Fast Path) pour les petits N (N <= 93)
// Pour les nombres qui tiennent dans un `uint64`, les opérations arithmétiques natives
// du CPU sont des ordres de grandeur plus rapides que l'arithmétique de `big.Int`.
// Cette fonction évite donc le coût de `big.Int` pour les cas les plus courants.
func (fd *OptimizedFastDoubling) calculateSmall(n uint64) *big.Int {
	if n == 0 {
		return big.NewInt(0)
	}
	var a, b uint64 = 0, 1
	// Calcul itératif simple.
	for i := uint64(1); i < n; i++ {
		a, b = b, a+b
	}
	// Le résultat est converti en `big.Int` uniquement à la fin.
	return new(big.Int).SetUint64(b)
}

// Calculate calcule le n-ième nombre de Fibonacci en utilisant l'algorithme "Fast Doubling".
// L'algorithme est basé sur les identités matricielles suivantes :
// F(2k)   = F(k) * [2*F(k+1) - F(k)]
// F(2k+1) = F(k+1)^2 + F(k)^2
//
// Cette méthode itère sur les bits de `n` du plus significatif (MSB) au moins
// significatif (LSB), appliquant ces formules pour "doubler" l'indice `k` à chaque étape.
func (fd *OptimizedFastDoubling) Calculate(ctx context.Context, progressChan chan<- float64, n uint64) (*big.Int, error) {

	// 1. Optimisation O1: Gérer les petits N
	// Si n <= 93, on utilise la version rapide qui n'utilise pas `big.Int`.
	if n <= MaxFibUint64 {
		reportProgress(progressChan, 1.0)
		return fd.calculateSmall(n), nil
	}

	// 2. Optimisation O2: Stratégie Zéro Allocation
	// On récupère notre objet `calculationState` depuis le pool.
	// `defer putState(s)` garantit qu'il sera retourné au pool à la fin de la fonction,
	// qu'il y ait une erreur ou non.
	s := getState()
	defer putState(s)

	// `bits.Len64(n)` donne le nombre de bits de `n`. C'est le nombre d'itérations nécessaires.
	numBits := bits.Len64(n)
	// Micro-optimisation : la multiplication est plus rapide que la division.
	// Nous pré-calculons l'inverse pour l'utiliser dans la boucle de progression.
	invNumBits := 1.0 / float64(numBits)

	// 3. Optimisation O3: Préparation à la parallélisation
	var wg sync.WaitGroup
	// On active la parallélisation uniquement si on a plus d'un cœur CPU.
	useParallel := runtime.NumCPU() > 1

	// 4. Boucle Principale (itération sur les bits de n)
	// On parcourt les bits de `n` de gauche à droite (du plus significatif au moins significatif).
	for i := numBits - 1; i >= 0; i-- {

		// Vérification périodique de l'annulation (via timeout ou Ctrl+C).
		// C'est crucial pour que le programme réponde rapidement à une demande d'arrêt.
		if ctx.Err() != nil {
			return nil, fmt.Errorf("calculation canceled: %w", ctx.Err())
		}

		// Reporter la progression (sauf pour la toute première itération).
		if progressChan != nil && i < numBits - 1 {
			progress := float64(numBits-1-i) * invNumBits
			reportProgress(progressChan, progress)
		}

		// --- Étape de Doublage (k -> 2k) ---
		// À chaque itération, on calcule F(2k) et F(2k+1) à partir de F(k) et F(k+1).

		// Calcul de F(2k) = F(k) * [2*F(k+1) - F(k)]
		// Cette partie est séquentielle car chaque opération dépend de la précédente.
		s.t1.Lsh(s.f_k1, 1)   // t1 = 2 * F(k+1)
		s.t2.Sub(s.t1, s.f_k) // t2 = t1 - F(k)
		s.t3.Mul(s.f_k, s.t2) // t3 = F(k) * t2  (futur F(2k))

		// Calcul de F(2k+1) = F(k+1)^2 + F(k)^2
		// Les deux mises au carré sont indépendantes et peuvent être parallélisées.

		// O3: Exécution parallèle si le nombre est assez grand et si on a plusieurs CPUs.
		if useParallel && s.f_k1.BitLen() > parallelThreshold {
			// `wg.Add(2)` indique que nous attendons la fin de 2 goroutines.
			wg.Add(2)

			// Goroutine 1: Calcule F(k+1)^2 et stocke le résultat dans `t1`.
			go func(dest, src *big.Int) {
				defer wg.Done()    // Signale la fin de cette goroutine.
				dest.Mul(src, src) // F(k+1) * F(k+1)
			}(s.t1, s.f_k1)

			// Goroutine 2: Calcule F(k)^2 et stocke le résultat dans `t4`.
			// On utilise `t4` pour ne pas interférer avec la goroutine 1 qui utilise `t1`.
			go func(dest, src *big.Int) {
				defer wg.Done()    // Signale la fin de cette goroutine.
				dest.Mul(src, src) // F(k) * F(k)
			}(s.t4, s.f_k)

			wg.Wait()              // Attend que les 2 goroutines soient terminées.
			s.f_k1.Add(s.t1, s.t4) // f_k1 = t1 + t4 (résultat final de F(2k+1))

		} else {
			// Exécution séquentielle pour les plus petits nombres ou sur un seul CPU.
			s.t1.Mul(s.f_k1, s.f_k1) // t1 = F(k+1)^2
			s.t4.Mul(s.f_k, s.f_k)   // t4 = F(k)^2
			s.f_k1.Add(s.t1, s.t4)   // f_k1 = t1 + t4
		}

		// Mise à jour de F(k) pour la prochaine itération.
		s.f_k.Set(s.t3) // F(k) devient F(2k)

		// --- Étape d'Addition (si le bit de `n` est à 1) ---
		// Si le i-ème bit de `n` est 1, cela signifie que `n` est impair à cette étape.
		// Nous devons donc avancer de k -> k+1, ce qui correspond à F(2k) -> F(2k+1).
		// Les formules pour cela sont :
		// F(k+1) = F(k) + F(k-1) -> ici, F(2k+2) = F(2k+1) + F(2k)
		// F(k)   = F(k-1)       -> ici, F(2k+1) = F(2k+1)
		if (n>>uint(i))&1 == 1 {
			// On utilise t1 pour sauvegarder l'ancienne valeur de f_k1
			s.t1.Set(s.f_k1) // t1 = F(2k+1)
			// f_k1 devient F(2k+2) = F(2k+1) + F(2k)
			s.f_k1.Add(s.f_k1, s.f_k)
			// f_k devient F(2k+1)
			s.f_k.Set(s.t1)
		}
	}

	reportProgress(progressChan, 1.0)

	// POINT CRITIQUE (O2): Renvoyer une COPIE du résultat.
	// L'objet `s.f_k` appartient au pool et sera réutilisé. Si nous renvoyions
	// `s.f_k` directement, l'appelant aurait une référence vers un objet qui
	// pourrait être modifié par un autre calcul. Nous créons donc une nouvelle
	// variable `big.Int` pour y copier le résultat final.
	return new(big.Int).Set(s.f_k), nil
}

// ----------------------------------------------------------------------------
// SECTION 2: LOGIQUE APPLICATIVE (Interface en Ligne de Commande - CLI)
// (Cette section gère l'interaction avec l'utilisateur, les paramètres et l'affichage)
// ----------------------------------------------------------------------------

// Codes de sortie standards pour les applications en ligne de commande.
const (
	ExitSuccess       = 0   // Le programme s'est terminé avec succès.
	ExitErrorGeneric  = 1   // Erreur non spécifique.
	ExitErrorTimeout  = 2   // Le calcul a dépassé le temps imparti.
	ExitErrorCanceled = 130 // L'utilisateur a annulé l'opération (Ctrl+C).
)

// AppConfig regroupe la configuration de l'application issue des arguments de la CLI.
type AppConfig struct {
	N       uint64        // L'indice 'n' à calculer.
	Verbose bool          // Faut-il afficher le résultat complet ?
	Timeout time.Duration // Délai maximum pour le calcul.
}

func main() {
	// Configuration de la CLI en utilisant le package `flag`.
	// La valeur par défaut de 'n' est élevée pour bien démontrer les optimisations.
	nFlag := flag.Uint64("n", 100000000, "L'indice 'n' de la séquence de Fibonacci à calculer.")
	verboseFlag := flag.Bool("v", false, "Affiche le résultat complet sans le tronquer.")
	timeoutFlag := flag.Duration("timeout", 5*time.Minute, "Délai maximum (ex: '30s', '1m', '1h').")
	flag.Parse()

	config := AppConfig{
		N:       *nFlag,
		Verbose: *verboseFlag,
		Timeout: *timeoutFlag,
	}

	// L'exécution est déléguée à la fonction `run`.
	// Cette séparation rend le code plus facile à tester : on peut tester `run`
	// sans avoir à simuler la ligne de commande et `os.Exit`.
	exitCode := run(context.Background(), config, os.Stdout)
	os.Exit(exitCode)
}

// run est le cœur de l'application CLI.
func run(ctx context.Context, config AppConfig, out io.Writer) int {
	var calculator Calculator = &OptimizedFastDoubling{}

	fmt.Fprintf(out, "Calcul de F(%d)...\n", config.N)
	fmt.Fprintf(out, "Algorithme : %s\n", calculator.Name())
	fmt.Fprintf(out, "Nombre de cœurs CPU disponibles : %d\n", runtime.NumCPU())
	fmt.Fprintf(out, "Timeout défini à %s.\n", config.Timeout)

	// --- Configuration du Contexte pour l'Annulation et le Timeout ---
	// Le `context` en Go est un mécanisme standard pour propager des signaux d'annulation,
	// des timeouts ou d'autres données à travers les appels de fonctions.
	//
	// 1. On crée un contexte avec un timeout. Si le calcul dépasse `config.Timeout`,
	//    le contexte sera automatiquement "annulé".
	ctx, cancelTimeout := context.WithTimeout(ctx, config.Timeout)
	defer cancelTimeout() // `defer` garantit que les ressources du timeout sont libérées.
	// 2. On enrichit ce contexte pour qu'il soit aussi annulé si l'utilisateur appuie
	//    sur Ctrl+C (SIGINT) ou si le système demande l'arrêt (SIGTERM).
	ctx, stopSignals := signal.NotifyContext(ctx, syscall.SIGINT, syscall.SIGTERM)
	defer stopSignals() // Libère les ressources associées à la notification des signaux.

	// --- Reporting de Progression ---
	// On crée un canal pour recevoir les mises à jour de la progression (de 0.0 à 1.0).
	progressChan := make(chan float64, 100) // Bufferisé pour ne pas bloquer l'émetteur.
	var wg sync.WaitGroup
	wg.Add(1)
	// La goroutine `displayProgress` écoutera sur ce canal et mettra à jour la barre de progression.
	go displayProgress(&wg, progressChan, out)

	// --- Exécution du Calcul ---
	startTime := time.Now()
	// On passe le `ctx` au calculateur. Celui-ci vérifiera périodiquement si le
	// contexte a été annulé pour arrêter le calcul prématurément si nécessaire.
	result, err := calculator.Calculate(ctx, progressChan, config.N)

	close(progressChan) // On ferme le canal pour signaler à `displayProgress` qu'il n'y aura plus de mises à jour.
	wg.Wait()           // On attend que `displayProgress` ait fini de s'afficher avant de continuer.
	duration := time.Since(startTime)

	// --- Gestion des Résultats et Erreurs ---
	fmt.Fprintln(out, "\n--- Résultats ---")
	if err != nil {
		return handleCalculationError(err, duration, config.Timeout, out)
	}

	displayResult(result, config.N, duration, config.Verbose, out)
	return ExitSuccess
}

// handleCalculationError utilise `errors.Is` pour une détection robuste des erreurs de contexte.
// C'est la manière moderne et fiable de vérifier les erreurs d'annulation en Go.
func handleCalculationError(err error, duration time.Duration, timeout time.Duration, out io.Writer) int {
	// Le calcul a-t-il été arrêté parce que le timeout a été atteint ?
	if errors.Is(err, context.DeadlineExceeded) {
		fmt.Fprintf(out, "Statut : Échec. Le calcul a dépassé le délai imparti (%s) après %s.\n", timeout, duration)
		return ExitErrorTimeout
	} else if errors.Is(err, context.Canceled) {
		// Le calcul a-t-il été arrêté par un signal externe (comme Ctrl+C) ?
		fmt.Fprintf(out, "Statut : Annulé (signal reçu ou Ctrl+C) après %s.\n", duration)
		return ExitErrorCanceled
	} else {
		// Toute autre erreur.
		fmt.Fprintf(out, "Statut : Échec. Erreur interne : %v\n", err)
		return ExitErrorGeneric
	}
}

// displayResult gère l'affichage du résultat final, y compris sa troncature.
func displayResult(result *big.Int, n uint64, duration time.Duration, verbose bool, out io.Writer) {
	fmt.Fprintf(out, "Statut : Succès\n")
	fmt.Fprintf(out, "Durée d'exécution : %s\n", duration)
	fmt.Fprintf(out, "Taille du résultat : %d bits.\n", result.BitLen())

	resultStr := result.String()
	numDigits := len(resultStr)
	fmt.Fprintf(out, "Nombre de chiffres décimaux : %d\n", numDigits)

	// Constantes pour la troncature du résultat si celui-ci est trop long.
	const truncationLimit = 50 // Au-delà de 50 chiffres, on tronque.
	const displayEdges = 20    // On affiche les 20 premiers et 20 derniers chiffres.

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
	// Un "ticker" se déclenche à intervalle régulier (ici, toutes les 100ms)
	// pour rafraîchir l'affichage même si on ne reçoit pas de nouvelle valeur de progression.
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	lastProgress := 0.0

	printBar := func(progress float64) {
		// Le caractère spécial `\r` (carriage return) déplace le curseur au début
		// de la ligne actuelle, sans passer à la ligne suivante.
		// La prochaine écriture va donc écraser la ligne précédente, créant
		// l'illusion d'une barre de progression qui se met à jour sur place.
		fmt.Fprintf(out, "\rProgression : %6.2f%% [%-30s]", progress*100, progressBar(progress, 30))
	}

	// Boucle principale qui attend des événements sur deux canaux.
	for {
		select {
		// Cas 1: On reçoit une nouvelle valeur de progression.
		case p, ok := <-progressChan:
			if !ok {
				// `ok` est `false` si le canal a été fermé. C'est le signal de fin.
				// On affiche inconditionnellement la barre à 100% pour garantir
				// que l'état final est toujours correctement affiché.
				printBar(1.0)
				fmt.Fprintln(out) // On passe à la ligne suivante pour ne pas écraser la barre finale.
				return
			}
			lastProgress = p // On stocke la dernière progression reçue.
		// Cas 2: Le ticker s'est déclenché.
		case <-ticker.C:
			// On rafraîchit la barre avec la dernière valeur connue.
			printBar(lastProgress)
		}
	}
}

// progressBar construit la chaîne de caractères représentant la barre de progression.
// L'utilisation de `strings.Builder` est une optimisation pour construire des chaînes
// de manière efficace, en évitant des allocations mémoires multiples.
func progressBar(progress float64, length int) string {
	if progress > 1.0 {
		progress = 1.0
	} else if progress < 0.0 {
		progress = 0.0
	}

	count := int(progress * float64(length))

	var builder strings.Builder
	builder.Grow(length) // Pré-alloue la mémoire pour la taille finale de la chaîne.

	for i := 0; i < length; i++ {
		if i < count {
			builder.WriteRune('■') // Caractère pour la partie remplie.
		} else {
			builder.WriteRune(' ') // Caractère pour la partie vide.
		}
	}
	return builder.String()
}
