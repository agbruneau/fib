// # Calculateur de Fibonacci : Une Étude de Cas en Haute Performance avec Go
//
// Ce programme implémente le calcul des nombres de la suite de Fibonacci (F(n))
// pour des indices `n` arbitrairement grands. Il sert de démonstration pédagogique
// sur la manière dont des algorithmes mathématiques avancés et des techniques
// d'optimisation modernes en Go peuvent résoudre des problèmes calculatoirement
// intensifs.
//
// ## Principes Fondamentaux et Défis Techniques
//
// 1. **La Complexité Algorithmique : De O(n) à O(log n) :**
//    L'algorithme itératif est en O(n). Ce programme implémente des algorithmes
//    en O(log n) opérations arithmétiques : "Fast Doubling" et "Exponentiation Matricielle".
//
// 2. **Le Véritable Goulot d'Étranglement (Coût Binaire) :**
//    Pour de grands `n`, F(n) a K bits. Le coût réel est O(log n * M(K)), où M(K)
//    est le coût de la multiplication de deux nombres de K bits. Go utilise des
//    algorithmes comme Karatsuba (O(K^1.58)). La performance est dominée par ces
//    multiplications de `big.Int`.
//
// ## Stratégies d'Optimisation Implémentées
//
// 1. **Parallélisme au Niveau des Opérations (Task Parallelism) :**
//    Les multiplications indépendantes à chaque étape sont exécutées simultanément
//    sur différents cœurs CPU via des goroutines. Un seuil (`parallelThreshold`)
//    est utilisé pour s'assurer que le gain dépasse l'overhead de synchronisation.
//
// 2. **Gestion Mémoire "Zéro-Allocation" et Pression GC :**
//    Utilisation intensive de `sync.Pool` pour recycler les structures de données
//    (`big.Int`), atteignant une "zéro-allocation" dans le chemin critique et
//    minimisant l'impact du Garbage Collector (GC).
//
// 3. **Optimisations Algorithmiques Spécifiques :**
//    - **Fast Path O(1) :** Utilisation d'une Table de Consultation (LUT) pré-calculée
//      pour n <= 93.
//    - **Symétrie Matricielle :** Réduction du nombre de multiplications de 8 à 4
//      lors de la mise au carré des matrices de Fibonacci.
//
// 4. **Concurrence Robuste et Gestion du Contexte :**
//    L'API `context` est utilisée pour une gestion propre des annulations (Ctrl+C)
//    et des timeouts dans un environnement concurrent complexe.

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
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"
)

// ----------------------------------------------------------------------------
// SECTION 1: MOTEURS DE CALCUL FIBONACCI (Théorie et Implémentation Optimisée)
// ----------------------------------------------------------------------------

const (
	// MaxFibUint64 est l'index N maximal pour lequel F(N) peut être représenté
	// par un `uint64`. F(93) est le dernier.
	MaxFibUint64 = 93

	// DefaultParallelThreshold est le seuil par défaut (en nombre de bits)
	// au-delà duquel la parallélisation des multiplications de `big.Int` est activée.
	// Concept Clé : Overhead vs Gain. Si le calcul est rapide (nombres petits),
	// l'overhead de création des goroutines et de synchronisation dépasse le gain.
	DefaultParallelThreshold = 2048
)

// --- Optimisation O(1) : Fast Path avec Lookup Table (LUT) ---

// fibLookupTable stocke les valeurs pré-calculées de F(0) à F(93).
// Nous stockons des pointeurs `*big.Int` pour correspondre au type de retour attendu.
var fibLookupTable [MaxFibUint64 + 1]*big.Int

// init est une fonction spéciale en Go, exécutée automatiquement avant main().
// Elle initialise la LUT de manière itérative (O(N)). Ce coût est payé une seule
// fois au démarrage du programme.
func init() {
	// Calcul itératif en utilisant l'arithmétique uint64 native pour la performance.
	var a, b uint64 = 0, 1
	for i := uint64(0); i <= MaxFibUint64; i++ {
		fibLookupTable[i] = new(big.Int).SetUint64(a)
		a, b = b, a+b
	}
}

// lookupSmall récupère le résultat depuis le cache LUT en O(1).
func lookupSmall(n uint64) *big.Int {
	// IMPORTANT : Nous retournons une COPIE de la valeur de la LUT.
	// Cela garantit l'immuabilité du cache, évitant que l'appelant ne modifie
	// les valeurs pré-calculées, ce qui causerait des bugs en accès concurrent.
	return new(big.Int).Set(fibLookupTable[n])
}

// --- Abstraction et Polymorphisme ---

// Calculator définit l'interface standard pour tout algorithme de Fibonacci.
// L'utilisation d'interfaces permet le polymorphisme.
type Calculator interface {
	// Calculate effectue le calcul F(n).
	// Le paramètre `threshold` est ajouté pour permettre l'ajustement dynamique du seuil de parallélisme.
	Calculate(ctx context.Context, progressChan chan<- float64, n uint64, threshold int) (*big.Int, error)
	Name() string
}

// coreCalculator est une interface interne représentant le cœur algorithmique.
type coreCalculator interface {
	CalculateCore(ctx context.Context, progressChan chan<- float64, n uint64, threshold int) (*big.Int, error)
	Name() string
}

// FibCalculator implémente le **Design Pattern Décorateur**.
// Il "enveloppe" un `coreCalculator` et lui ajoute des fonctionnalités transversales
// (le "fast path" O(1)), respectant ainsi le principe de Responsabilité Unique (SRP).
type FibCalculator struct {
	core coreCalculator
}

func (c *FibCalculator) Name() string {
	return c.core.Name()
}

// Calculate est le point d'entrée unifié. Il agit comme un dispatcher.
func (c *FibCalculator) Calculate(ctx context.Context, progressChan chan<- float64, n uint64, threshold int) (*big.Int, error) {
	// OPTIMISATION : Fast Path O(1)
	if n <= MaxFibUint64 {
		reportProgress(progressChan, 1.0)
		return lookupSmall(n), nil
	}

	// Cas complexe : Délégation au calculateur de cœur O(log n).
	return c.core.CalculateCore(ctx, progressChan, n, threshold)
}

// reportProgress effectue un envoi non bloquant sur le canal de progression.
// Concept Clé : Communication Concurrente Non Bloquante.
// L'idiome `select` avec `default` garantit que si le récepteur (l'interface
// utilisateur) est lent, la mise à jour est ignorée au lieu de bloquer la
// goroutine de calcul.
func reportProgress(progressChan chan<- float64, progress float64) {
	if progressChan == nil {
		return
	}
	select {
	case progressChan <- progress:
	default: // Canal plein ou non prêt. On ignore la mise à jour.
	}
}

// ----------------------------------------------------------------------------
// IMPLÉMENTATION 1: ALGORITHME "OPTIMIZED FAST DOUBLING" (O(log n))
// ----------------------------------------------------------------------------
//
// Le "Fast Doubling" est généralement l'algorithme le plus rapide en pratique.
//
// ## Fondements Mathématiques
//
// L'algorithme repose sur les identités suivantes :
//
// 1. **F(2k)   = F(k) * [2*F(k+1) - F(k)]**
// 2. **F(2k+1) = F(k+1)² + F(k)²**
//
// Cela permet de calculer (F(2k), F(2k+1)) à partir de (F(k), F(k+1)) en seulement
// 3 multiplications distinctes.
//
// ## Logique de l'Algorithme (Itération Binaire Top-Down)
//
// On parcourt la représentation binaire de `n` du MSB vers le LSB.
// On maintient un état (F(k), F(k+1)).
// Pour chaque bit :
// 1. **Doubling :** L'état passe à (F(2k), F(2k+1)).
// 2. **Addition :** Si le bit est 1, l'état passe à (F(2k+1), F(2k+2)).

// --- Optimisation Mémoire : Pooling et Zéro-Allocation ---

// calculationState regroupe tous les `big.Int` nécessaires pour une itération.
type calculationState struct {
	f_k, f_k1      *big.Int // F(k), F(k+1)
	t1, t2, t3, t4 *big.Int // Variables temporaires
}

// statePool est une implémentation de `sync.Pool`.
// Concept Clé : Réduction de la Pression sur le Garbage Collector (GC).
// `sync.Pool` est un cache d'objets thread-safe. Au lieu d'allouer et de laisser
// le GC nettoyer ces objets coûteux, nous les réutilisons.
var statePool = sync.Pool{
	// La fonction New est appelée lorsque le pool est vide.
	New: func() interface{} {
		// Allocation initiale des objets `big.Int`.
		return &calculationState{
			f_k: new(big.Int), f_k1: new(big.Int),
			t1: new(big.Int), t2: new(big.Int),
			t3: new(big.Int), t4: new(big.Int),
		}
	},
}

// getState récupère un état depuis le pool et l'initialise (F(0)=0, F(1)=1).
func getState() *calculationState {
	s := statePool.Get().(*calculationState)
	s.f_k.SetInt64(0)
	s.f_k1.SetInt64(1)
	return s
}

// putState retourne l'état au pool.
func putState(s *calculationState) {
	statePool.Put(s)
}

// OptimizedFastDoubling est la structure implémentant l'algorithme.
type OptimizedFastDoubling struct{}

func (fd *OptimizedFastDoubling) Name() string {
	return "OptimizedFastDoubling (3-Way-Parallel+ZeroAlloc+LUT)"
}

// CalculateCore est le cœur de l'implémentation du Fast Doubling.
func (fd *OptimizedFastDoubling) CalculateCore(ctx context.Context, progressChan chan<- float64, n uint64, threshold int) (*big.Int, error) {
	// Récupération d'un état recyclé. `defer` garantit son retour au pool.
	s := getState()
	defer putState(s)

	// Pré-calcul pour la boucle.
	numBits := bits.Len64(n) // Nombre d'itérations O(log n).
	invNumBits := 1.0 / float64(numBits)

	// Configuration du parallélisme.
	var wg sync.WaitGroup
	// Le parallélisme n'est activé que si le CPU a plus d'un cœur logique.
	useParallel := runtime.NumCPU() > 1

	// Boucle principale : Parcours des bits de n (MSB vers LSB).
	for i := numBits - 1; i >= 0; i-- {

		// --- Gestion de l'Annulation ---
		// Vérification proactive du contexte à chaque itération.
		if ctx.Err() != nil {
			return nil, fmt.Errorf("calculation canceled: %w", ctx.Err())
		}
		// Rapport de progression (non bloquant).
		if progressChan != nil && i < numBits-1 {
			reportProgress(progressChan, float64(numBits-1-i)*invNumBits)
		}

		// --- Étape 1: Doubling (Calcul de F(2k) et F(2k+1)) ---

		// 1.1 Calcul du terme commun : (2*F(k+1) - F(k))
		s.t2.Lsh(s.f_k1, 1)   // t2 = F(k+1) * 2 (Décalage de bit efficace)
		s.t2.Sub(s.t2, s.f_k) // t2 = 2*F(k+1) - F(k)

		// 1.2 Calcul des 3 multiplications indépendantes :
		// A. F(k) * t2
		// B. F(k+1)²
		// C. F(k)²

		// OPTIMISATION : Parallélisme Conditionnel (Task Parallelism)
		if useParallel && s.f_k1.BitLen() > threshold {
			// Lancement de 3 goroutines (Fan-out).
			wg.Add(3)

			// Goroutine A
			go func(dest, src1, src2 *big.Int) {
				defer wg.Done()
				dest.Mul(src1, src2)
			}(s.t3, s.f_k, s.t2) // t3 = F(k) * t2

			// Goroutine B
			go func(dest, src *big.Int) {
				defer wg.Done()
				dest.Mul(src, src)
			}(s.t1, s.f_k1) // t1 = F(k+1)²

			// Goroutine C
			go func(dest, src *big.Int) {
				defer wg.Done()
				dest.Mul(src, src)
			}(s.t4, s.f_k) // t4 = F(k)²

			// Point de synchronisation (Barrière / Fan-in).
			wg.Wait()

			// Assemblage final.
			s.f_k.Set(s.t3)        // f_k devient F(2k)
			s.f_k1.Add(s.t1, s.t4) // f_k1 devient F(2k+1)

		} else {
			// Exécution séquentielle.
			s.t3.Mul(s.f_k, s.t2)
			s.t1.Mul(s.f_k1, s.f_k1)
			s.t4.Mul(s.f_k, s.f_k)
			s.f_k.Set(s.t3)
			s.f_k1.Add(s.t1, s.t4)
		}

		// --- Étape 2: Addition Conditionnelle ---
		// Si le i-ème bit de n est 1, nous devons avancer l'état d'un cran.
		if (n>>uint(i))&1 == 1 {
			// (F(k), F(k+1)) devient (F(k+1), F(k)+F(k+1)).
			s.t1.Set(s.f_k1)
			s.f_k1.Add(s.f_k1, s.f_k)
			s.f_k.Set(s.t1)
		}
	}

	reportProgress(progressChan, 1.0)
	// Le résultat F(n) est dans f_k. Nous retournons une copie car l'original
	// sera retourné au pool et potentiellement modifié par un autre calcul.
	return new(big.Int).Set(s.f_k), nil
}

// ----------------------------------------------------------------------------
// IMPLÉMENTATION 2: ALGORITHME D'EXPONENTIATION MATRICIELLE (O(log n))
// ----------------------------------------------------------------------------
//
// Cette approche utilise l'algèbre linéaire. Elle est basée sur la matrice de
// Fibonacci Q = [[1, 1], [1, 0]]. L'identité clé est que Q^n contient F(n).
//
// ## Algorithme : Exponentiation par Carré (Binary Exponentiation)
//
// On calcule Q^n en O(log n) étapes en décomposant n en binaire (ex: Q^13 = Q^8 * Q^4 * Q^1).
//
// ## Optimisation Spécifique : La Symétrie
//
// Q est symétrique. Toute puissance de Q est symétrique ([[a, b], [b, d]]).
// La mise au carré ne nécessite que 4 multiplications au lieu de 8 :
//   [[a²+b², b*(a+d)], [b*(a+d), b²+d²]]

// matrix représente une matrice 2x2 de `big.Int`.
type matrix struct {
	a, b, c, d *big.Int // [[a, b], [c, d]]
}

// newMatrix est une fonction utilitaire pour allouer une nouvelle matrice initialisée.
func newMatrix() *matrix {
	return &matrix{new(big.Int), new(big.Int), new(big.Int), new(big.Int)}
}

// Set copie les valeurs de `other` dans `m`.
func (m *matrix) Set(other *matrix) {
	m.a.Set(other.a)
	m.b.Set(other.b)
	m.c.Set(other.c)
	m.d.Set(other.d)
}

// --- Optimisation Mémoire : Pooling pour les Matrices ---

// matrixState contient toutes les structures nécessaires pour l'algorithme matriciel.
// OPTIMISATION : `tempMatrix` a été ajoutée ici pour assurer une véritable
// "zéro-allocation", en évitant son allocation dans CalculateCore.
type matrixState struct {
	res        *matrix // Matrice résultat (accumulateur R).
	p          *matrix // Matrice de puissance courante (P).
	tempMatrix *matrix // Matrice temporaire pour les calculs intermédiaires.

	// t1-t8: Variables temporaires pour les multiplications.
	t1, t2, t3, t4, t5, t6, t7, t8 *big.Int
}

var matrixStatePool = sync.Pool{
	New: func() interface{} {
		// Allocation initiale de tous les `big.Int` (20 au total).
		return &matrixState{
			res:        newMatrix(),
			p:          newMatrix(),
			tempMatrix: newMatrix(),
			t1:         new(big.Int), t2: new(big.Int), t3: new(big.Int), t4: new(big.Int),
			t5: new(big.Int), t6: new(big.Int), t7: new(big.Int), t8: new(big.Int),
		}
	},
}

// getMatrixState récupère et initialise un état depuis le pool.
func getMatrixState() *matrixState {
	s := matrixStatePool.Get().(*matrixState)
	// Initialisation de res à la Matrice Identité I = [[1, 0], [0, 1]].
	s.res.a.SetInt64(1)
	s.res.b.SetInt64(0)
	s.res.c.SetInt64(0)
	s.res.d.SetInt64(1)
	// Initialisation de p à la Matrice Q = [[1, 1], [1, 0]].
	s.p.a.SetInt64(1)
	s.p.b.SetInt64(1)
	s.p.c.SetInt64(1)
	s.p.d.SetInt64(0)
	return s
}

func putMatrixState(s *matrixState) {
	matrixStatePool.Put(s)
}

// MatrixExponentiation est la structure implémentant l'algorithme.
type MatrixExponentiation struct{}

func (me *MatrixExponentiation) Name() string {
	return "MatrixExponentiation (SymmetricOpt+Parallel+ZeroAlloc+LUT)"
}

// squareSymmetricMatrix implémente l'optimisation de la mise au carré symétrique (4 multiplications).
func squareSymmetricMatrix(dest, m *matrix, s *matrixState, useParallel bool, threshold int) {
	var wg sync.WaitGroup

	// Utilisation des temporaires du pool.
	t_a_sq := s.t1     // a²
	t_b_sq := s.t2     // b²
	t_d_sq := s.t3     // d²
	t_b_ad := s.t4     // b*(a+d)
	t_a_plus_d := s.t5 // (a+d)

	t_a_plus_d.Add(m.a, m.d)

	// OPTIMISATION : Parallélisme Conditionnel (4-Way Parallelism)
	if useParallel && m.a.BitLen() > threshold {
		wg.Add(4)
		go func() { defer wg.Done(); t_a_sq.Mul(m.a, m.a) }()
		go func() { defer wg.Done(); t_b_sq.Mul(m.b, m.b) }()
		go func() { defer wg.Done(); t_d_sq.Mul(m.d, m.d) }()
		go func() { defer wg.Done(); t_b_ad.Mul(m.b, t_a_plus_d) }()
		wg.Wait() // Barrière de synchronisation.
	} else {
		// Exécution séquentielle.
		t_a_sq.Mul(m.a, m.a)
		t_b_sq.Mul(m.b, m.b)
		t_d_sq.Mul(m.d, m.d)
		t_b_ad.Mul(m.b, t_a_plus_d)
	}

	// Assemblage final.
	dest.a.Add(t_a_sq, t_b_sq) // a²+b²
	dest.b.Set(t_b_ad)         // b*(a+d)
	dest.c.Set(t_b_ad)         // Symétrie : c = b
	dest.d.Add(t_b_sq, t_d_sq) // b²+d²
}

// multiplyMatrices effectue la multiplication générique `dest = m1 * m2` (8 multiplications).
func multiplyMatrices(dest, m1, m2 *matrix, s *matrixState, useParallel bool, threshold int) {
	var wg sync.WaitGroup

	// OPTIMISATION : Parallélisme Conditionnel (8-Way Parallelism)
	if useParallel && m1.a.BitLen() > threshold {
		wg.Add(8)
		// Lancement des 8 goroutines pour chaque produit partiel.
		go func() { defer wg.Done(); s.t1.Mul(m1.a, m2.a) }() // a1*a2
		go func() { defer wg.Done(); s.t2.Mul(m1.b, m2.c) }() // b1*c2
		go func() { defer wg.Done(); s.t3.Mul(m1.a, m2.b) }() // a1*b2
		go func() { defer wg.Done(); s.t4.Mul(m1.b, m2.d) }() // b1*d2
		go func() { defer wg.Done(); s.t5.Mul(m1.c, m2.a) }() // c1*a2
		go func() { defer wg.Done(); s.t6.Mul(m1.d, m2.c) }() // d1*c2
		go func() { defer wg.Done(); s.t7.Mul(m1.c, m2.b) }() // c1*b2
		go func() { defer wg.Done(); s.t8.Mul(m1.d, m2.d) }() // d1*d2
		wg.Wait()                                             // Barrière de synchronisation.
	} else {
		// Exécution séquentielle.
		s.t1.Mul(m1.a, m2.a)
		s.t2.Mul(m1.b, m2.c)
		s.t3.Mul(m1.a, m2.b)
		s.t4.Mul(m1.b, m2.d)
		s.t5.Mul(m1.c, m2.a)
		s.t6.Mul(m1.d, m2.c)
		s.t7.Mul(m1.c, m2.b)
		s.t8.Mul(m1.d, m2.d)
	}
	// Assemblage final (4 additions).
	dest.a.Add(s.t1, s.t2)
	dest.b.Add(s.t3, s.t4)
	dest.c.Add(s.t5, s.t6)
	dest.d.Add(s.t7, s.t8)
}

// CalculateCore est le cœur de l'implémentation de l'Exponentiation Matricielle.
func (me *MatrixExponentiation) CalculateCore(ctx context.Context, progressChan chan<- float64, n uint64, threshold int) (*big.Int, error) {
	// Pour n>0, on calcule Q^(n-1). Le cas n=0 est géré par le Fast Path.
	// Bien que le décorateur gère n<=93, on ajoute une sécurité ici pour n=0.
	if n == 0 {
		return big.NewInt(0), nil
	}

	s := getMatrixState()
	defer putMatrixState(s)

	k := n - 1 // L'exposant cible.
	numBits := bits.Len64(k)
	invNumBits := 1.0 / float64(numBits)
	useParallel := runtime.NumCPU() > 1

	// Utilisation de la matrice temporaire issue du pool (Optimisation Zéro-Alloc).
	tempMatrix := s.tempMatrix

	// --- Algorithme d'Exponentiation par Carré (Bottom-Up) ---
	// Parcours des bits de k (LSB vers MSB).
	for i := 0; i < numBits; i++ {
		// Gestion de l'annulation et progression.
		if ctx.Err() != nil {
			return nil, fmt.Errorf("calculation canceled: %w", ctx.Err())
		}
		if progressChan != nil {
			reportProgress(progressChan, float64(i)*invNumBits)
		}

		// ÉTAPE 1: Multiplication conditionnelle (Si le bit est 1).
		// res = res * p
		if (k>>uint(i))&1 == 1 {
			multiplyMatrices(tempMatrix, s.res, s.p, s, useParallel, threshold)
			s.res.Set(tempMatrix) // Copie du résultat dans l'accumulateur.
		}

		// ÉTAPE 2: Mise au carré pour l'itération suivante.
		// p = p * p (Passe de Q^(2^i) à Q^(2^(i+1)))
		// Utilisation de l'optimisation symétrique.
		squareSymmetricMatrix(tempMatrix, s.p, s, useParallel, threshold)
		s.p.Set(tempMatrix)
	}

	reportProgress(progressChan, 1.0)
	// À la fin, res = Q^(n-1). F(n) est le coefficient (0,0).
	// Retourne une copie.
	return new(big.Int).Set(s.res.a), nil
}

// ----------------------------------------------------------------------------
// SECTION 2: LOGIQUE DE L'APPLICATION (CLI, Orchestration et Concurrence)
// ----------------------------------------------------------------------------
//
// Cette section illustre le principe de **Séparation des Préoccupations (SoC)**.
// Elle gère l'interface utilisateur et l'orchestration, en restant découplée
// des moteurs de calcul (Section 1).

// Codes de sortie standards pour l'intégration système.
const (
	ExitSuccess       = 0
	ExitErrorGeneric  = 1   // Erreur de configuration ou interne.
	ExitErrorTimeout  = 2   // Délai dépassé.
	ExitErrorCanceled = 130 // Annulation par l'utilisateur (Ctrl+C), convention shell.
	ExitErrorMismatch = 3   // Incohérence des résultats en mode comparaison.
)

// AppConfig regroupe les paramètres de l'application.
type AppConfig struct {
	N         uint64
	Verbose   bool
	Timeout   time.Duration
	Algo      string
	Threshold int // Seuil de parallélisme configurable.
}

// ProgressUpdate structure utilisée pour la communication concurrente de la progression.
type ProgressUpdate struct {
	CalculatorIndex int
	Value           float64
}

// calculatorRegistry centralise les algorithmes disponibles (Extensibilité).
var calculatorRegistry = map[string]Calculator{
	"fast":   &FibCalculator{core: &OptimizedFastDoubling{}},
	"matrix": &FibCalculator{core: &MatrixExponentiation{}},
}

// main est le point d'entrée du programme.
func main() {
	// Configuration des arguments de ligne de commande (CLI).
	nFlag := flag.Uint64("n", 100000000, "L'indice 'n' de la séquence de Fibonacci à calculer.")
	verboseFlag := flag.Bool("v", false, "Affiche le résultat complet (peut être très long).")
	timeoutFlag := flag.Duration("timeout", 5*time.Minute, "Délai maximum (ex: 30s, 1m).")
	algoFlag := flag.String("algo", "all", "Algorithme : 'fast', 'matrix', ou 'all' (comparaison).")
	// Flag pour le réglage fin du seuil de parallélisme.
	thresholdFlag := flag.Int("threshold", DefaultParallelThreshold, "Seuil (en bits) pour activer la multiplication parallèle. Optimisation bas niveau.")

	flag.Parse()

	config := AppConfig{
		N:         *nFlag,
		Verbose:   *verboseFlag,
		Timeout:   *timeoutFlag,
		Algo:      *algoFlag,
		Threshold: *thresholdFlag,
	}

	// Exécution de la logique principale et sortie avec le code approprié.
	exitCode := run(context.Background(), config, os.Stdout)
	os.Exit(exitCode)
}

// CalculationResult stocke le résultat d'une exécution d'algorithme.
type CalculationResult struct {
	Name     string
	Result   *big.Int
	Duration time.Duration
	Err      error
}

// run est la fonction d'orchestration principale.
func run(ctx context.Context, config AppConfig, out io.Writer) int {

	// --- Gestion Robuste du Contexte et de l'Annulation ---
	// Mise en place d'un contexte multi-sources pour l'annulation.

	// 1. Timeout : Annulation automatique après un délai.
	ctx, cancelTimeout := context.WithTimeout(ctx, config.Timeout)
	defer cancelTimeout() // Libération des ressources du timer.

	// 2. Signaux OS : Annulation sur Ctrl+C (SIGINT) ou demande de terminaison (SIGTERM).
	ctx, stopSignals := signal.NotifyContext(ctx, syscall.SIGINT, syscall.SIGTERM)
	defer stopSignals() // Nettoyage de l'écoute des signaux.

	// Affichage de la configuration initiale.
	fmt.Fprintf(out, "--- Configuration ---\n")
	fmt.Fprintf(out, "Calcul de F(%d).\n", config.N)
	fmt.Fprintf(out, "Système : CPU Cores=%d | Go Runtime=%s\n", runtime.NumCPU(), runtime.Version())
	fmt.Fprintf(out, "Paramètres : Timeout=%s | Parallel Threshold=%d bits\n", config.Timeout, config.Threshold)

	// --- Sélection des Calculateurs ---
	var calculatorsToRun []Calculator
	algo := strings.ToLower(config.Algo)

	if algo == "all" {
		// Mode comparaison : Exécution parallèle de tous les algorithmes.
		fmt.Fprintf(out, "Mode : Comparaison (Exécution parallèle de %d algorithmes).\n", len(calculatorRegistry))
		// Tri des clés pour un ordre d'affichage déterministe.
		keys := make([]string, 0, len(calculatorRegistry))
		for k := range calculatorRegistry {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			calculatorsToRun = append(calculatorsToRun, calculatorRegistry[k])
		}
	} else {
		// Mode simple.
		calculator, ok := calculatorRegistry[algo]
		if !ok {
			fmt.Fprintf(out, "Erreur : algorithme '%s' inconnu.\n", config.Algo)
			return ExitErrorGeneric
		}
		fmt.Fprintf(out, "Mode : Simple exécution.\nAlgorithme : %s\n", calculator.Name())
		calculatorsToRun = append(calculatorsToRun, calculator)
	}

	fmt.Fprintf(out, "\n--- Exécution ---\n")

	// --- Exécution et Analyse ---
	results := executeCalculations(ctx, calculatorsToRun, config, out)

	if len(results) == 1 {
		// Analyse du résultat unique.
		res := results[0]
		fmt.Fprintln(out, "\n--- Résultat Final ---")
		if res.Err != nil {
			return handleCalculationError(res.Err, res.Duration, config.Timeout, out)
		}
		displayResult(res.Result, config.N, res.Duration, config.Verbose, out)
		return ExitSuccess
	}

	// Analyse des résultats de comparaison.
	return analyzeComparisonResults(results, config, out)
}

// executeCalculations orchestre l'exécution parallèle des calculateurs.
// Concept Clé : Modèle de Concurrence "Fan-Out/Fan-In" avec pipeline de progression.
// Cette architecture garantit une terminaison propre (Graceful Shutdown).
func executeCalculations(ctx context.Context, calculators []Calculator, config AppConfig, out io.Writer) []CalculationResult {
	results := make([]CalculationResult, len(calculators))

	// --- Synchronisation Complexe : Gestion des Cycles de Vie des Goroutines ---
	// 3 WaitGroups sont nécessaires pour coordonner les étapes et assurer une
	// fermeture propre des canaux (évitant les paniques "send on closed channel").

	// 1. calcWg: Attend la fin des calculs principaux.
	// 2. proxyWg: Attend la fin des goroutines de relais de progression.
	// 3. displayWg: Attend la fin de la goroutine d'affichage.
	var calcWg, proxyWg, displayWg sync.WaitGroup

	// Canal central d'agrégation de la progression (Fan-In).
	progressChan := make(chan ProgressUpdate, len(calculators)*10) // Bufferisé.

	// Lancement de la goroutine d'affichage (Consommateur).
	displayWg.Add(1)
	go displayAggregateProgress(&displayWg, progressChan, len(calculators), out)

	// Lancement des goroutines de calcul (Producteurs - Fan-Out).
	for i, calc := range calculators {
		calcWg.Add(1)
		go func(idx int, calculator Calculator) {
			defer calcWg.Done()

			// Canal "proxy" dédié à ce calcul.
			proxyChan := make(chan float64, 100)

			// Goroutine "Proxy" : Relais et Enrichissement.
			proxyWg.Add(1)
			go func() {
				defer proxyWg.Done()
				// La boucle `range` se termine lorsque `proxyChan` est fermé.
				for p := range proxyChan {
					// Envoi sécurisé vers le canal central.
					select {
					case progressChan <- ProgressUpdate{CalculatorIndex: idx, Value: p}:
					case <-ctx.Done(): // Sécurité si le contexte est annulé pendant l'envoi.
						return
					}
				}
			}()

			startTime := time.Now()
			// Appel du calcul principal. On passe le seuil configuré.
			res, err := calculator.Calculate(ctx, proxyChan, config.N, config.Threshold)

			// Fermeture du canal proxy. Signale la fin de la production de progression.
			close(proxyChan)

			results[idx] = CalculationResult{
				Name:     calculator.Name(),
				Result:   res,
				Duration: time.Since(startTime),
				Err:      err,
			}
		}(i, calc)
	}

	// --- Séquence de Fermeture (Coordination Finale) ---
	// L'ordre est critique pour une terminaison propre.

	// 1. Attendre la fin de tous les calculs.
	calcWg.Wait()
	// 2. Attendre que tous les proxys aient fini de relayer.
	proxyWg.Wait()
	// 3. Fermer le canal central `progressChan`. C'est sûr car plus aucun producteur n'est actif.
	close(progressChan)
	// 4. Attendre que l'affichage soit terminé.
	displayWg.Wait()

	return results
}

// analyzeComparisonResults valide l'intégrité des résultats en mode "all".
func analyzeComparisonResults(results []CalculationResult, config AppConfig, out io.Writer) int {
	fmt.Fprintln(out, "\n--- Résultats de la Comparaison (Benchmark & Validation) ---")

	var firstResult *big.Int
	var firstError error
	successCount := 0

	// Affichage des statistiques individuelles.
	for _, res := range results {
		status := "Succès"
		if res.Err != nil {
			status = fmt.Sprintf("Échec (%v)", res.Err)
			if firstError == nil {
				firstError = res.Err
			}
		} else {
			successCount++
			if firstResult == nil {
				firstResult = res.Result
			}
		}
		// Formatage aligné pour une meilleure lisibilité.
		fmt.Fprintf(out, "  - %-65s | Durée: %-15s | Statut: %s\n", res.Name, res.Duration.String(), status)
	}

	// Gestion des cas d'échec global.
	if successCount == 0 {
		fmt.Fprintln(out, "\nStatut Global : Échec. Tous les calculs ont échoué.")
		return handleCalculationError(firstError, 0, config.Timeout, out)
	}

	// Validation de la Cohérence (Test d'intégrité critique).
	mismatch := false
	for _, res := range results {
		// Comparaison des résultats valides avec le premier résultat valide.
		// Utilisation de Cmp (0 signifie égalité).
		if res.Err == nil && res.Result.Cmp(firstResult) != 0 {
			mismatch = true
			break
		}
	}

	if mismatch {
		fmt.Fprintln(out, "\nStatut Global : Échec Critique ! Les algorithmes ont produit des résultats différents.")
		return ExitErrorMismatch
	}

	fmt.Fprintln(out, "\nStatut Global : Succès. Tous les résultats valides sont identiques.")
	displayResult(firstResult, config.N, 0, config.Verbose, out)
	return ExitSuccess
}

// handleCalculationError interprète les erreurs et détermine le code de sortie.
func handleCalculationError(err error, duration time.Duration, timeout time.Duration, out io.Writer) int {
	msg := ""
	if duration > 0 {
		msg = fmt.Sprintf(" après %s", duration)
	}

	// Utilisation de `errors.Is` pour vérifier les erreurs de contexte (enveloppées).
	// C'est l'idiome Go moderne pour la gestion des erreurs de contexte.
	if errors.Is(err, context.DeadlineExceeded) {
		// Timeout.
		fmt.Fprintf(out, "Statut : Échec (Timeout). Le délai imparti (%s) a été dépassé%s.\n", timeout, msg)
		return ExitErrorTimeout
	} else if errors.Is(err, context.Canceled) {
		// Annulation manuelle (Ctrl+C ou signal).
		fmt.Fprintf(out, "Statut : Annulé (Signal reçu)%s.\n", msg)
		return ExitErrorCanceled
	} else {
		// Autre erreur interne.
		fmt.Fprintf(out, "Statut : Échec. Erreur interne : %v\n", err)
		return ExitErrorGeneric
	}
}

// displayResult formate et affiche le résultat final F(n).
func displayResult(result *big.Int, n uint64, duration time.Duration, verbose bool, out io.Writer) {
	fmt.Fprintln(out, "\n--- Données du Résultat ---")
	if duration > 0 {
		// Affiche la durée seulement si elle provient d'un calcul unique.
		fmt.Fprintf(out, "Durée d'exécution : %s\n", duration)
	}

	// Métadonnées sur la taille du nombre.
	bitLen := result.BitLen()
	fmt.Fprintf(out, "Taille Binaire : %d bits.\n", bitLen)

	// Calcul du nombre de chiffres décimaux (conversion en chaîne).
	resultStr := result.String() // Conversion en base 10.
	numDigits := len(resultStr)
	fmt.Fprintf(out, "Nombre de Chiffres Décimaux : %d\n", numDigits)

	// Affichage du résultat (Gestion de la Troncature).
	const truncationLimit = 80
	const displayEdges = 20

	if verbose {
		// Mode verbeux : Affichage complet.
		fmt.Fprintf(out, "\nF(%d) = %s\n", n, resultStr)
	} else if numDigits > truncationLimit {
		// Troncature pour les très grands nombres.
		fmt.Fprintf(out, "F(%d) (Tronqué) = %s...%s\n", n, resultStr[:displayEdges], resultStr[numDigits-displayEdges:])
	} else {
		// Affichage complet pour les nombres courts.
		fmt.Fprintf(out, "F(%d) = %s\n", n, resultStr)
	}
}

// displayAggregateProgress gère l'affichage dynamique de la barre de progression.
func displayAggregateProgress(wg *sync.WaitGroup, progressChan <-chan ProgressUpdate, numCalculators int, out io.Writer) {
	defer wg.Done()
	progresses := make([]float64, numCalculators)

	// OPTIMISATION : Taux de Rafraîchissement Limité (100ms / 10Hz).
	// Utilisation d'un `time.Ticker` pour éviter de surcharger le terminal.
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	// Fonction interne pour dessiner la barre de progression.
	printBar := func() {
		// Calcul de la progression moyenne.
		var totalProgress float64
		for _, p := range progresses {
			totalProgress += p
		}
		avgProgress := totalProgress / float64(numCalculators)

		label := "Progression"
		if numCalculators > 1 {
			label = "Progression Moyenne"
		}
		// Le caractère `\r` (Retour Chariot) permet de réécrire sur la même ligne.
		fmt.Fprintf(out, "\r%s : %6.2f%% [%-30s]", label, avgProgress*100, progressBar(avgProgress, 30))
	}

	// Boucle d'événements principale (Pattern Select).
	for {
		select {
		// Cas 1: Réception d'une mise à jour.
		case update, ok := <-progressChan:
			// Si `ok` est faux, le canal est fermé (signal de terminaison).
			if !ok {
				// Affichage final à 100%.
				for i := range progresses {
					progresses[i] = 1.0
				}
				printBar()
				fmt.Fprintln(out) // Passage à la ligne final.
				return
			}
			// Mise à jour de l'état interne.
			if update.CalculatorIndex < len(progresses) {
				progresses[update.CalculatorIndex] = update.Value
			}

		// Cas 2: Déclenchement du Ticker.
		case <-ticker.C:
			// Redessine la barre.
			printBar()
		}
	}
}

// progressBar génère la représentation textuelle de la barre.
func progressBar(progress float64, length int) string {
	if progress > 1.0 {
		progress = 1.0
	} else if progress < 0.0 {
		progress = 0.0
	}
	count := int(progress * float64(length))
	var builder strings.Builder
	// Optimisation : Pré-allocation de la mémoire de la chaîne.
	builder.Grow(length)
	for i := 0; i < length; i++ {
		if i < count {
			builder.WriteRune('■') // Caractère de remplissage.
		} else {
			builder.WriteRune(' ')
		}
	}
	return builder.String()
}
