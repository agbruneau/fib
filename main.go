// # Calculateur de Fibonacci Hautes Performances
//
// Ce programme Go est une implémentation à but pédagogique et de benchmark pour
// le calcul de très grands nombres de la suite de Fibonacci. Il met en œuvre deux
// des algorithmes les plus efficaces connus pour ce problème, tous deux avec une
// complexité temporelle de O(log n), ce qui les rend capables de calculer des
// termes comme F(100,000,000) en quelques secondes seulement.
//
// ## Objectifs Pédagogiques et Techniques
//
// 1. **Vulgarisation d'Algorithmes Avancés :**
//    - **Optimized Fast Doubling :** Une méthode très rapide qui utilise des
//      identités mathématiques spécifiques à Fibonacci pour doubler l'indice `k`
//      à chaque étape (calculant F(2k) et F(2k+1) à partir de F(k) et F(k+1)).
//    - **Exponentiation Matricielle :** Une approche plus générale qui repose sur
//      le fait que F(n) peut être obtenu en élevant une matrice spécifique à la
//      puissance n.
//
// 2. **Démonstration d'Optimisations en Go :**
//    - **Parallélisme :** Le code tire parti des processeurs multi-cœurs en
//      parallélisant les multiplications de grands nombres (`*big.Int`), qui sont
//      les opérations les plus coûteuses.
//    - **Gestion de la Mémoire "Zéro-Allocation" :** Pour éviter la pression sur le
//      ramasse-miettes (Garbage Collector) lors des calculs intensifs, le programme
//      utilise des "pools" d'objets (`sync.Pool`). Les structures de données
//      temporaires sont recyclées au lieu d'être constamment créées et détruites.
//    - **Gestion du Contexte :** L'API `context` de Go est utilisée pour gérer
//      proprement l'annulation des calculs (par exemple, sur un Ctrl+C de
//      l'utilisateur) et pour imposer un délai maximum (timeout).
//
// 3. **Interface en Ligne de Commande (CLI) Robuste :**
//    - Le programme fournit une interface simple pour choisir l'algorithme,
//      spécifier le nombre `n`, activer un mode verbeux et définir un timeout.
//    - Un mode "comparaison" permet de lancer les deux algorithmes en parallèle
//      pour vérifier que leurs résultats sont identiques, agissant comme un test
//      de validité dynamique.
//
// ## Structure du Fichier
//
// - **SECTION 1: MOTEURS DE CALCUL FIBONACCI :** Contient la logique pure des
//   algorithmes. C'est le cœur du programme, entièrement découplé de l'interface
//   utilisateur.
// - **SECTION 2: LOGIQUE DE L'APPLICATION :** Gère l'interface en ligne de
//   commande, l'affichage de la progression, l'interprétation des résultats et
//   la gestion des erreurs.
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
// SECTION 1: MOTEURS DE CALCUL FIBONACCI (Logique principale, hautement optimisée)
// ----------------------------------------------------------------------------

const (
	// MaxFibUint64 est l'index N le plus élevé pour lequel le résultat F(N) peut
	// être contenu dans un entier non signé de 64 bits (uint64).
	// F(93) est le dernier de la série qui ne dépasse pas 2^64 - 1.
	// Au-delà, l'utilisation de `math/big.Int` est indispensable.
	// Le calculer et le stocker en constante permet une optimisation "fast path".
	MaxFibUint64 = 93

	// parallelThreshold définit la taille minimale (en bits) d'un grand nombre
	// (`big.Int`) au-delà de laquelle la parallélisation de la multiplication
	// devient rentable.
	// En dessous de ce seuil, le coût de la création de goroutines et de la
	// synchronisation (overhead) est supérieur au gain de temps du calcul parallèle.
	// Cette valeur a été déterminée empiriquement.
	parallelThreshold = 2048
)

// Calculator est une interface qui définit un "contrat" pour tout algorithme
// de calcul de Fibonacci. Cela permet à la logique de l'application (Section 2)
// de traiter indifféremment les différentes implémentations (Fast Doubling,
// Matrix Exponentiation, etc.), les rendant interchangeables.
type Calculator interface {
	// Calculate est la méthode principale qui effectue le calcul pour un n donné.
	// - ctx: Le contexte permet de gérer l'annulation (timeout, signal utilisateur).
	// - progressChan: Un canal pour envoyer des mises à jour de la progression (de 0.0 à 1.0).
	// - n: L'index dans la suite de Fibonacci à calculer.
	Calculate(ctx context.Context, progressChan chan<- float64, n uint64) (*big.Int, error)

	// Name retourne le nom descriptif de l'algorithme.
	Name() string
}

// coreCalculator est une interface interne pour le cœur d'un algorithme de
// Fibonacci, sans la logique commune (comme le "fast path").
type coreCalculator interface {
	CalculateCore(ctx context.Context, progressChan chan<- float64, n uint64) (*big.Int, error)
	Name() string
}

// FibCalculator est un décorateur qui prend un `coreCalculator` et lui ajoute
// la logique commune, comme l'optimisation "fast path". Il implémente
// l'interface publique `Calculator`.
type FibCalculator struct {
	core coreCalculator
}

// Name retourne le nom du calculateur de cœur.
func (c *FibCalculator) Name() string {
	return c.core.Name()
}

// Calculate exécute le calcul. Il gère d'abord le cas rapide (n <= 93)
// et délègue au calculateur de cœur pour les grands nombres.
func (c *FibCalculator) Calculate(ctx context.Context, progressChan chan<- float64, n uint64) (*big.Int, error) {
	if n <= MaxFibUint64 {
		reportProgress(progressChan, 1.0)
		return calculateSmall(n), nil
	}
	return c.core.CalculateCore(ctx, progressChan, n)
}

// reportProgress envoie une valeur de progression dans le canal `progressChan`
// sans jamais bloquer l'exécution.
// Si le canal est plein (parce que le récepteur est lent), la mise à jour est
// simplement ignorée. Ceci est crucial pour ne pas ralentir le calcul lui-même.
// L'instruction `select` avec une clause `default` est l'idiome Go classique
// pour un envoi non bloquant sur un canal.
func reportProgress(progressChan chan<- float64, progress float64) {
	if progressChan == nil {
		return
	}
	select {
	case progressChan <- progress:
	default: // Ne fait rien si le canal n'est pas prêt à recevoir.
	}
}

// calculateSmall est une fonction d'optimisation (fast path) pour les "petits"
// nombres de Fibonacci (n <= 93), ceux dont le résultat tient dans un `uint64`.
// Pour ces cas, un simple algorithme itératif est beaucoup plus rapide que les
// algorithmes complexes (log n) car il n'implique pas la surcharge liée à
// l'allocation de mémoire pour les `big.Int` ou à la complexité de l'algorithme.
func calculateSmall(n uint64) *big.Int {
	if n == 0 {
		return big.NewInt(0)
	}
	// Itération classique pour calculer F(n).
	var a, b uint64 = 0, 1
	for i := uint64(1); i < n; i++ {
		a, b = b, a+b // L'assignation multiple évite une variable temporaire.
	}
	// Le résultat est converti en `*big.Int` pour correspondre au type de
	// retour de l'interface `Calculator`.
	return new(big.Int).SetUint64(b)
}

// ----------------------------------------------------------------------------
// IMPLÉMENTATION 1: ALGORITHME "OPTIMIZED FAST DOUBLING"
// ----------------------------------------------------------------------------
//
// Le "Fast Doubling" est l'un des algorithmes les plus rapides pour calculer
// les nombres de Fibonacci. Sa complexité est O(log n).
//
// Principe mathématique :
// L'algorithme s'appuie sur les deux identités suivantes :
//   F(2k)   = F(k) * [2*F(k+1) - F(k)]
//   F(2k+1) = F(k+1)^2 + F(k)^2
//
// L'idée est de parcourir les bits de l'entier `n` du plus significatif au
// moins significatif. On maintient constamment le couple (F(k), F(k+1)).
//
// À chaque étape, on effectue une opération de "doubling" :
//   (F(k), F(k+1)) -> (F(2k), F(2k+1))
//
// Si le bit de `n` actuellement lu est 1, cela signifie que notre cible est
// impaire. On doit donc effectuer une étape supplémentaire d'"addition" pour
// avancer de 2k à 2k+1 :
//   (F(2k), F(2k+1)) -> (F(2k+1), F(2k+2))
//
// En répétant ce processus pour tous les bits de `n`, on arrive au résultat F(n).

// calculationState contient tous les grands entiers nécessaires pour une passe
// de l'algorithme Fast Doubling.
// Le fait de regrouper ces valeurs dans une structure permet de les recycler
// facilement via un `sync.Pool`, évitant ainsi des allocations mémoire coûteuses
// à chaque appel de la fonction de calcul.
// f_k:  Stocke F(k)
// f_k1: Stocke F(k+1)
// t1-t4: Nombres temporaires utilisés pour les calculs intermédiaires.
type calculationState struct {
	f_k, f_k1, t1, t2, t3, t4 *big.Int
}

// statePool est un "pool" d'objets `calculationState`.
// `sync.Pool` est un mécanisme de cache mémoire fourni par Go. Il permet de
// réutiliser des objets qui ne sont plus utilisés au lieu de les laisser au
// ramasse-miettes (GC) et d'en allouer de nouveaux.
// C'est une optimisation critique pour les fonctions appelées fréquemment ou
// manipulant des objets lourds, comme c'est le cas ici.
var statePool = sync.Pool{
	// La fonction New est appelée par le pool si aucun objet n'est disponible.
	// Elle crée une nouvelle instance de `calculationState` avec tous les
	// `big.Int` déjà initialisés, prêts à l'emploi.
	New: func() interface{} {
		return &calculationState{
			f_k: new(big.Int), f_k1: new(big.Int),
			t1: new(big.Int), t2: new(big.Int),
			t3: new(big.Int), t4: new(big.Int),
		}
	},
}

// getState récupère un `calculationState` depuis le pool.
// Il réinitialise les valeurs initiales de la suite, F(0)=0 et F(1)=1,
// pour que le calcul puisse commencer.
func getState() *calculationState {
	s := statePool.Get().(*calculationState)
	s.f_k.SetInt64(0)  // F(0)
	s.f_k1.SetInt64(1) // F(1)
	return s
}

// putState remet un `calculationState` dans le pool une fois qu'il n'est
// plus utilisé, le rendant disponible pour un prochain calcul.
func putState(s *calculationState) {
	statePool.Put(s)
}

// OptimizedFastDoubling est l'implémentation de l'interface Calculator.
type OptimizedFastDoubling struct{}

// Name retourne le nom de l'algorithme, incluant les optimisations clés.
func (fd *OptimizedFastDoubling) Name() string {
	return "OptimizedFastDoubling (3-Way-Parallel+ZeroAlloc)"
}

// CalculateCore exécute le cœur de l'algorithme Fast Doubling pour F(n).
// La logique du "fast path" est gérée par le FibCalculator qui l'enveloppe.
func (fd *OptimizedFastDoubling) CalculateCore(ctx context.Context, progressChan chan<- float64, n uint64) (*big.Int, error) {
	// Récupération d'un état depuis le pool. `defer` garantit qu'il sera
	// retourné au pool à la fin de la fonction, même en cas d'erreur.
	s := getState()
	defer putState(s)

	numBits := bits.Len64(n) // Nombre de bits dans la représentation binaire de n.
	invNumBits := 1.0 / float64(numBits)
	var wg sync.WaitGroup
	useParallel := runtime.NumCPU() > 1

	// Boucle sur les bits de n, du plus significatif (MSB) au moins significatif (LSB).
	for i := numBits - 1; i >= 0; i-- {
		// Vérification de l'annulation du contexte (timeout ou Ctrl+C).
		if ctx.Err() != nil {
			return nil, fmt.Errorf("calculation canceled: %w", ctx.Err())
		}
		if progressChan != nil && i < numBits-1 {
			reportProgress(progressChan, float64(numBits-1-i)*invNumBits)
		}

		// --- Étape de "Doubling" ---
		// Calcule F(2k) et F(2k+1) à partir de F(k) et F(k+1).
		// F(2k)   = F(k) * (2*F(k+1) - F(k))
		// F(2k+1) = F(k+1)^2 + F(k)^2

		// Calcul de (2*F(k+1) - F(k))
		s.t2.Lsh(s.f_k1, 1)    // t2 = F(k+1) * 2
		s.t2.Sub(s.t2, s.f_k) // t2 = 2*F(k+1) - F(k)

		// Les trois multiplications suivantes sont les opérations les plus coûteuses.
		// Si les nombres sont assez grands et que plusieurs cœurs CPU sont
		// disponibles, elles sont exécutées en parallèle.
		if useParallel && s.f_k1.BitLen() > parallelThreshold {
			wg.Add(3)
			// t3 = F(k) * t2  (le futur F(2k))
			go func(dest, src1, src2 *big.Int) { defer wg.Done(); dest.Mul(src1, src2) }(s.t3, s.f_k, s.t2)
			// t1 = F(k+1)^2
			go func(dest, src *big.Int) { defer wg.Done(); dest.Mul(src, src) }(s.t1, s.f_k1)
			// t4 = F(k)^2
			go func(dest, src *big.Int) { defer wg.Done(); dest.Mul(src, src) }(s.t4, s.f_k)
			wg.Wait() // Attente de la fin des 3 multiplications.
			s.f_k.Set(s.t3)            // f_k devient F(2k)
			s.f_k1.Add(s.t1, s.t4)     // f_k1 devient F(k+1)^2 + F(k)^2 = F(2k+1)
		} else {
			// Exécution séquentielle pour les nombres plus petits.
			s.t3.Mul(s.f_k, s.t2)      // t3 = F(k) * (2*F(k+1) - F(k))
			s.t1.Mul(s.f_k1, s.f_k1)   // t1 = F(k+1)^2
			s.t4.Mul(s.f_k, s.f_k)     // t4 = F(k)^2
			s.f_k.Set(s.t3)            // f_k devient F(2k)
			s.f_k1.Add(s.t1, s.t4)     // f_k1 devient F(2k+1)
		}

		// --- Étape d'"Addition" ---
		// Si le i-ème bit de n est 1, on doit avancer l'état de F(2k) à F(2k+1).
		// L'état (F(k), F(k+1)) devient (F(k+1), F(k) + F(k+1)).
		if (n>>uint(i))&1 == 1 {
			// Avant : f_k = F(2k), f_k1 = F(2k+1)
			s.t1.Set(s.f_k1)               // t1 sauvegarde F(2k+1)
			s.f_k1.Add(s.f_k1, s.f_k)      // f_k1 = F(2k+1) + F(2k) = F(2k+2)
			s.f_k.Set(s.t1)                // f_k = t1 = F(2k+1)
			// Après : f_k = F(2k+1), f_k1 = F(2k+2)
		}
	}

	reportProgress(progressChan, 1.0)
	// Le résultat final est F(n), qui se trouve dans f_k.
	return new(big.Int).Set(s.f_k), nil
}

// ----------------------------------------------------------------------------
// IMPLÉMENTATION 2: ALGORITHME D'EXPONENTIATION MATRICIELLE
// ----------------------------------------------------------------------------
//
// Cette approche, de complexité O(log n), est une méthode élégante qui utilise
// l'algèbre linéaire pour calculer les nombres de Fibonacci. Elle est souvent un
// peu moins performante que le "Fast Doubling" en pratique à cause d'un plus
// grand nombre de multiplications, mais elle reste un exemple classique et
// puissant de la façon dont un problème récursif peut être transformé en un
// problème d'exponentiation.
//
// 1. LE PRINCIPE MATHÉMATIQUE : LA MATRICE DE FIBONACCI
// ---------------------------------------------------------
// La relation de récurrence F(n+1) = F(n) + F(n-1) peut être exprimée sous
// forme matricielle. L'idée est de trouver une matrice 2x2, appelée Q, qui,
// lorsqu'elle est multipliée par un vecteur contenant deux termes consécutifs
// de la suite, nous donne le vecteur des deux termes suivants.
//
// On cherche à passer de [ F(n), F(n-1) ] à [ F(n+1), F(n) ].
//
//  [ F(n+1) ] = [ F(n) + F(n-1) ]
//  [ F(n)   ] = [ F(n)          ]
//
// Ceci peut se réécrire comme une multiplication de matrice :
//
//  [ F(n+1) ]   [ 1  1 ] [ F(n)   ]
//  [        ] = [      ] [        ]
//  [ F(n)   ]   [ 1  0 ] [ F(n-1) ]
//
// La matrice Q = [[1, 1], [1, 0]] est la "matrice de Fibonacci".
//
// En appliquant cette transformation n fois à partir de l'état initial [F(1), F(0)] = [1, 0],
// on obtient une formule générale :
//
//  [ F(n+1)  F(n)   ]   [ 1  1 ]^n   (soit Q^n)
//  [                ] = [      ]
//  [ F(n)    F(n-1) ]   [ 1  0 ]
//
// Pour calculer F(n), il nous suffit donc d'élever la matrice Q à la puissance n.
// Le résultat F(n) sera le coefficient en haut à droite (ou en bas à gauche) de la matrice résultante.
// Note : ce programme calcule Q^(n-1) et prend le coefficient en haut à gauche, ce qui est équivalent.
//
//
// 2. L'OPTIMISATION : L'EXPONENTIATION PAR CARRÉ (BINARY EXPONENTIATION)
// ---------------------------------------------------------------------
// Calculer Q^n en multipliant Q par lui-même n-1 fois serait inefficace (O(n)).
// On utilise plutôt l'exponentiation par carré, un algorithme en O(log n).
//
// L'idée est de décomposer l'exposant `n` en sa représentation binaire.
// Par exemple, pour calculer x^13 :
// L'exposant 13 en binaire est 1101 (8 + 4 + 0 + 1).
// Donc, x^13 = x^(8+4+1) = x^8 * x^4 * x^1.
//
// L'algorithme fonctionne ainsi :
//   a. On commence avec un résultat (res = 1) et une puissance de x (p = x).
//   b. On parcourt les bits de l'exposant de droite à gauche.
//   c. Si le bit est 1, on multiplie le résultat par la puissance courante : res = res * p.
//   d. On met la puissance au carré pour le bit suivant : p = p * p.
//
// Exemple pour x^13 (1101):
// - bit 0 (1): res = 1 * x^1 = x      | p devient x^2
// - bit 1 (0): res = x                | p devient x^4
// - bit 2 (1): res = x * x^4 = x^5    | p devient x^8
// - bit 3 (1): res = x^5 * x^8 = x^13 | p devient x^16
//
// Ce principe s'applique à l'identique pour les matrices. On calcule Q^n en
// n'effectuant que des multiplications et des mises au carré de matrices.
//
//
// 3. OPTIMISATION SPÉCIFIQUE : LA SYMETRIE DE LA MATRICE
// ---------------------------------------------------------
// La matrice Q est symétrique (Q[i,j] = Q[j,i]). Une propriété utile est que
// le produit de matrices symétriques n'est pas toujours symétrique, mais la
// *puissance* d'une matrice symétrique l'est toujours.
//
// Donc, toutes les matrices `p` (les puissances de Q) dans l'algorithme
// d'exponentiation seront symétriques. Une matrice symétrique 2x2 a la forme :
//
//   [ a  b ]
//   [ b  d ]
//
// La mise au carré d'une telle matrice donne :
//
//   [ a^2 + b^2    a*b + b*d ]   [ a^2 + b^2    b*(a+d) ]
//   [ b*a + d*b    b^2 + d^2 ] = [ b*(a+d)    b^2 + d^2 ]
//
// Le calcul de ce carré ne requiert que 4 multiplications (a*a, b*b, d*d, b*(a+d))
// au lieu des 8 nécessaires pour une multiplication de matrices génériques.
// Cette optimisation, implémentée dans `squareSymmetricMatrix`, divise presque
// par deux le coût de chaque étape de mise au carré.

// matrix représente une matrice 2x2 de grands entiers.
// [ a  b ]
// [ c  d ]
type matrix struct {
	a, b, c, d *big.Int
}

// Set copie les valeurs d'une autre matrice dans la matrice réceptrice (m).
func (m *matrix) Set(other *matrix) {
	m.a.Set(other.a)
	m.b.Set(other.b)
	m.c.Set(other.c)
	m.d.Set(other.d)
}

// matrixState contient tous les objets nécessaires pour le calcul par
// exponentiation matricielle. Comme pour `calculationState`, cette structure
// est mise en pool avec `sync.Pool` pour recycler la mémoire.
// res: La matrice résultat, initialisée à la matrice identité.
// p:   La matrice de Fibonacci [[1,1],[1,0]] qui sera élevée au carré à chaque étape.
// t1-t8: Entiers temporaires pour les 8 multiplications requises.
type matrixState struct {
	res, p                         *matrix
	t1, t2, t3, t4, t5, t6, t7, t8 *big.Int
}

// matrixStatePool est le pool d'objets pour `matrixState`.
var matrixStatePool = sync.Pool{
	New: func() interface{} {
		// Alloue une seule fois tous les big.Int nécessaires.
		return &matrixState{
			res: &matrix{new(big.Int), new(big.Int), new(big.Int), new(big.Int)},
			p:   &matrix{new(big.Int), new(big.Int), new(big.Int), new(big.Int)},
			t1:  new(big.Int), t2: new(big.Int), t3: new(big.Int), t4: new(big.Int),
			t5: new(big.Int), t6: new(big.Int), t7: new(big.Int), t8: new(big.Int),
		}
	},
}

// getMatrixState récupère un état depuis le pool et l'initialise.
func getMatrixState() *matrixState {
	s := matrixStatePool.Get().(*matrixState)
	// res = Matrice Identité I
	s.res.a.SetInt64(1)
	s.res.b.SetInt64(0)
	s.res.c.SetInt64(0)
	s.res.d.SetInt64(1)
	// p = Matrice de Fibonacci Q
	s.p.a.SetInt64(1)
	s.p.b.SetInt64(1)
	s.p.c.SetInt64(1)
	s.p.d.SetInt64(0)
	return s
}

// putMatrixState remet un état dans le pool pour réutilisation.
func putMatrixState(s *matrixState) {
	matrixStatePool.Put(s)
}

// MatrixExponentiation est l'implémentation de l'interface Calculator.
type MatrixExponentiation struct{}

// Name retourne le nom de l'algorithme, incluant les optimisations clés.
func (me *MatrixExponentiation) Name() string {
	return "MatrixExponentiation (SymmetricOpt+Parallel+ZeroAlloc)"
}

// squareSymmetricMatrix calcule `dest = m * m` où `m` est une matrice symétrique.
// Une matrice symétrique est de la forme [[a, b], [b, d]]. La mettre au carré
// résulte en [[a*a+b*b, b*(a+d)], [b*(a+d), b*b+d*d]].
//
// Ce calcul peut être effectué avec seulement 4 multiplications de `big.Int` au
// lieu des 8 requises pour une multiplication de matrices générique, ce qui
// accélère considérablement le processus. C'est l'optimisation la plus
// importante de cet algorithme.
func squareSymmetricMatrix(dest, m *matrix, s *matrixState, useParallel bool) {
	var wg sync.WaitGroup

	// Nous avons besoin de 4 multiplications : a*a, b*b, d*d, et b*(a+d).
	// Nous utilisons les entiers temporaires du pool `matrixState`.
	t_a_sq := s.t1
	t_b_sq := s.t2
	t_d_sq := s.t3
	t_b_ad := s.t4
	t_a_plus_d := s.t5 // Pour la somme intermédiaire a+d

	t_a_plus_d.Add(m.a, m.d)

	if useParallel && m.a.BitLen() > parallelThreshold {
		wg.Add(4)
		go func() { defer wg.Done(); t_a_sq.Mul(m.a, m.a) }()         // a^2
		go func() { defer wg.Done(); t_b_sq.Mul(m.b, m.b) }()         // b^2
		go func() { defer wg.Done(); t_d_sq.Mul(m.d, m.d) }()         // d^2
		go func() { defer wg.Done(); t_b_ad.Mul(m.b, t_a_plus_d) }() // b*(a+d)
		wg.Wait()
	} else {
		t_a_sq.Mul(m.a, m.a)
		t_b_sq.Mul(m.b, m.b)
		t_d_sq.Mul(m.d, m.d)
		t_b_ad.Mul(m.b, t_a_plus_d)
	}

	// Assemblage des coefficients de la matrice finale à partir des résultats.
	// dest.a = a^2 + b^2
	dest.a.Add(t_a_sq, t_b_sq)
	// dest.b = b*(a+d)
	dest.b.Set(t_b_ad)
	// dest.c = dest.b car la matrice résultante est aussi symétrique.
	dest.c.Set(t_b_ad)
	// dest.d = b^2 + d^2
	dest.d.Add(t_b_sq, t_d_sq)
}

// multiplyMatrices effectue la multiplication générique `dest = m1 * m2`.
//
// Formule de multiplication pour deux matrices 2x2 :
//  [ a1  b1 ] * [ a2  b2 ] = [ a1*a2+b1*c2  a1*b2+b1*d2 ]
//  [ c1  d1 ]   [ c2  d2 ]   [ c1*a2+d1*c2  c1*b2+d1*d2 ]
//
// Elle utilise les entiers temporaires du `matrixState` pour stocker les
// résultats intermédiaires des 8 multiplications requises.
// Celles-ci sont parallélisées si `useParallel` est vrai et que la taille
// des nombres dépasse `parallelThreshold`.
func multiplyMatrices(dest, m1, m2 *matrix, s *matrixState, useParallel bool) {
	var wg sync.WaitGroup
	if useParallel && m1.a.BitLen() > parallelThreshold {
		wg.Add(8)
		go func() { defer wg.Done(); s.t1.Mul(m1.a, m2.a) }() // m1.a*m2.a
		go func() { defer wg.Done(); s.t2.Mul(m1.b, m2.c) }() // m1.b*m2.c
		go func() { defer wg.Done(); s.t3.Mul(m1.a, m2.b) }() // m1.a*m2.b
		go func() { defer wg.Done(); s.t4.Mul(m1.b, m2.d) }() // m1.b*m2.d
		go func() { defer wg.Done(); s.t5.Mul(m1.c, m2.a) }() // m1.c*m2.a
		go func() { defer wg.Done(); s.t6.Mul(m1.d, m2.c) }() // m1.d*m2.c
		go func() { defer wg.Done(); s.t7.Mul(m1.c, m2.b) }() // m1.c*m2.b
		go func() { defer wg.Done(); s.t8.Mul(m1.d, m2.d) }() // m1.d*m2.d
		wg.Wait()
	} else {
		// Exécution séquentielle
		s.t1.Mul(m1.a, m2.a)
		s.t2.Mul(m1.b, m2.c)
		s.t3.Mul(m1.a, m2.b)
		s.t4.Mul(m1.b, m2.d)
		s.t5.Mul(m1.c, m2.a)
		s.t6.Mul(m1.d, m2.c)
		s.t7.Mul(m1.c, m2.b)
		s.t8.Mul(m1.d, m2.d)
	}
	// Les 4 additions finales pour obtenir les coefficients de la matrice résultat.
	dest.a.Add(s.t1, s.t2) // a = t1+t2
	dest.b.Add(s.t3, s.t4) // b = t3+t4
	dest.c.Add(s.t5, s.t6) // c = t5+t6
	dest.d.Add(s.t7, s.t8) // d = t7+t8
}

// CalculateCore exécute le cœur de l'algorithme pour F(n) via l'exponentiation matricielle.
// La logique du "fast path" (n <= 93) est gérée par le FibCalculator qui l'enveloppe.
func (me *MatrixExponentiation) CalculateCore(ctx context.Context, progressChan chan<- float64, n uint64) (*big.Int, error) {
	// La vérification n=0 est gérée par le fast path. Pour la formule matricielle,
	// on calcule Q^(n-1).
	s := getMatrixState()
	defer putMatrixState(s)

	k := n - 1 // L'exposant est n-1.
	numBits := bits.Len64(k)
	invNumBits := 1.0 / float64(numBits)
	useParallel := runtime.NumCPU() > 1

	// `tempMatrix` est nécessaire pour stocker le résultat d'une multiplication
	// avant de l'assigner à la matrice de destination, pour éviter de corrompre
	// les données sources en cours de calcul (ex: p = p * p).
	tempMatrix := &matrix{new(big.Int), new(big.Int), new(big.Int), new(big.Int)}

	// --- Algorithme d'exponentiation par carré (binaire) ---
	// On parcourt les bits de l'exposant k, du moins significatif (LSB) au plus
	// significatif (MSB).
	// `s.res` est notre accumulateur de résultat, initialisé à la matrice identité.
	// `s.p` est la puissance de la matrice Q, initialisée à Q^1.
	for i := 0; i < numBits; i++ {
		if ctx.Err() != nil {
			return nil, fmt.Errorf("calculation canceled: %w", ctx.Err())
		}
		if progressChan != nil {
			reportProgress(progressChan, float64(i)*invNumBits)
		}

		// ÉTAPE 1: Test du bit courant.
		// Si le i-ème bit de k est à 1, on doit inclure la puissance actuelle
		// de la matrice dans notre résultat final.
		// Ex: pour k=13 (1101), on le fait pour i=0, i=2, i=3.
		if (k>>uint(i))&1 == 1 {
			// res = res * p
			multiplyMatrices(tempMatrix, s.res, s.p, s, useParallel)
			s.res.Set(tempMatrix) // Copie du résultat dans l'accumulateur.
		}

		// ÉTAPE 2: Mise au carré pour l'itération suivante.
		// La matrice p passe de Q^(2^i) à Q^(2^(i+1)).
		// p = p * p
		// On utilise la fonction optimisée `squareSymmetricMatrix` car p (une
		// puissance de la matrice symétrique Q) est toujours symétrique.
		squareSymmetricMatrix(tempMatrix, s.p, s, useParallel)
		s.p.Set(tempMatrix) // Copie du résultat.
	}

	reportProgress(progressChan, 1.0)
	// À la fin, res = Q^(n-1). Le résultat F(n) est le coefficient (0,0)
	// (en haut à gauche) de cette matrice.
	return new(big.Int).Set(s.res.a), nil
}

// ----------------------------------------------------------------------------
// SECTION 2: LOGIQUE DE L'APPLICATION (Interface en ligne de commande)
// ----------------------------------------------------------------------------
//
// Cette section contient tout le code qui n'est pas directement lié aux
// algorithmes de Fibonacci. Elle gère l'interaction avec l'utilisateur,
// l'affichage, et l'orchestration des calculs.

// Définition des codes de sortie du programme, pour une utilisation dans des scripts.
const (
	ExitSuccess       = 0   // Le programme s'est terminé avec succès.
	ExitErrorGeneric  = 1   // Erreur générique (ex: mauvais argument).
	ExitErrorTimeout  = 2   // Le calcul a dépassé le délai imparti.
	ExitErrorCanceled = 130 // Le calcul a été annulé par l'utilisateur (Ctrl+C). Convention shell.
	ExitErrorMismatch = 3   // En mode comparaison, les résultats des algorithmes ne correspondent pas.
)

// AppConfig regroupe tous les paramètres de configuration de l'application
// issus de la ligne de commande.
type AppConfig struct {
	N       uint64        // L'indice du nombre de Fibonacci à calculer.
	Verbose bool          // Si vrai, affiche le nombre complet.
	Timeout time.Duration // Délai maximum pour le calcul.
	Algo    string        // Nom de l'algorithme à utiliser ('fast', 'matrix', ou 'all').
}

// ProgressUpdate est utilisée par le mode comparaison pour rapporter la
// progression de chaque calculateur individuel à un gestionnaire central.
type ProgressUpdate struct {
	CalculatorIndex int     // Index du calculateur dans la liste.
	Value           float64 // Progression (0.0 à 1.0).
}

// calculatorRegistry centralise la liste des algorithmes de calcul disponibles.
// L'utilisation d'un registre permet d'ajouter de nouveaux algorithmes en ne
// les déclarant qu'à un seul endroit. La clé est le nom utilisé dans l'argument
// de ligne de commande --algo.
var calculatorRegistry = map[string]Calculator{
	"fast":   &FibCalculator{core: &OptimizedFastDoubling{}},
	"matrix": &FibCalculator{core: &MatrixExponentiation{}},
}

// main est le point d'entrée du programme.
func main() {
	// Configuration et analyse des arguments de la ligne de commande avec le package `flag`.
	nFlag := flag.Uint64("n", 100000000, "L'indice 'n' de la séquence de Fibonacci à calculer.")
	verboseFlag := flag.Bool("v", false, "Affiche le résultat complet.")
	timeoutFlag := flag.Duration("timeout", 5*time.Minute, "Délai maximum (ex: 30s, 1m).")
	algoFlag := flag.String("algo", "all", "Algorithme : 'fast', 'matrix', ou 'all' (par défaut) pour comparer.")
	flag.Parse() // Analyse les arguments fournis.

	// Création d'une structure de configuration pour passer les paramètres proprement.
	config := AppConfig{
		N:       *nFlag,
		Verbose: *verboseFlag,
		Timeout: *timeoutFlag,
		Algo:    *algoFlag,
	}

	// Appel de la fonction `run` qui contient la logique principale de l'application.
	// `os.Stdout` est passé pour que `run` écrive sur la sortie standard.
	// Le code de sortie de `run` est utilisé pour terminer le programme.
	exitCode := run(context.Background(), config, os.Stdout)
	os.Exit(exitCode)
}

// CalculationResult stocke le résultat d'un seul calcul, incluant la durée et
// une potentielle erreur.
type CalculationResult struct {
	Name     string
	Result   *big.Int
	Duration time.Duration
	Err      error
}

// run est la fonction d'orchestration principale.
// Elle configure le contexte, sélectionne les calculateurs, les exécute,
// puis analyse et affiche les résultats.
func run(ctx context.Context, config AppConfig, out io.Writer) int {
	ctx, cancelTimeout := context.WithTimeout(ctx, config.Timeout)
	defer cancelTimeout()
	ctx, stopSignals := signal.NotifyContext(ctx, syscall.SIGINT, syscall.SIGTERM)
	defer stopSignals()

	fmt.Fprintf(out, "Calcul de F(%d)...\n", config.N)
	fmt.Fprintf(out, "Nombre de cœurs CPU disponibles : %d\n", runtime.NumCPU())
	fmt.Fprintf(out, "Timeout défini à %s.\n", config.Timeout)

	// --- Sélection des calculateurs ---
	var calculatorsToRun []Calculator
	algo := strings.ToLower(config.Algo)

	if algo == "all" {
		fmt.Fprintf(out, "Mode comparaison : Lancement de %d algorithmes en parallèle...\n", len(calculatorRegistry))
		// Ordonner les clés du registre pour garantir un affichage stable.
		keys := make([]string, 0, len(calculatorRegistry))
		for k := range calculatorRegistry {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			calculatorsToRun = append(calculatorsToRun, calculatorRegistry[k])
		}
	} else {
		calculator, ok := calculatorRegistry[algo]
		if !ok {
			availableAlgos := make([]string, 0, len(calculatorRegistry))
			for k := range calculatorRegistry {
				availableAlgos = append(availableAlgos, "'"+k+"'")
			}
			fmt.Fprintf(out, "Erreur : algorithme '%s' inconnu. Choix possibles : %s, ou 'all'.\n",
				config.Algo, strings.Join(availableAlgos, ", "))
			return ExitErrorGeneric
		}
		fmt.Fprintf(out, "Algorithme : %s\n", calculator.Name())
		calculatorsToRun = append(calculatorsToRun, calculator)
	}

	// --- Exécution ---
	results := executeCalculations(ctx, calculatorsToRun, config, out)

	// --- Analyse et affichage des résultats ---
	if len(results) == 1 {
		res := results[0]
		fmt.Fprintln(out, "\n--- Résultats ---")
		if res.Err != nil {
			return handleCalculationError(res.Err, res.Duration, config.Timeout, out)
		}
		displayResult(res.Result, config.N, res.Duration, config.Verbose, out)
		return ExitSuccess
	}
	return analyzeComparisonResults(results, config, out)
}

// executeCalculations exécute un ou plusieurs calculateurs en parallèle et gère l'affichage de la progression.
func executeCalculations(ctx context.Context, calculators []Calculator, config AppConfig, out io.Writer) []CalculationResult {
	results := make([]CalculationResult, len(calculators))
	// On a besoin de 3 WaitGroups pour synchroniser correctement les différentes étapes :
	// 1. calcWg: pour attendre la fin des calculs principaux.
	// 2. proxyWg: pour attendre que les goroutines intermédiaires (proxy) aient fini de rapporter la progression.
	// 3. displayWg: pour attendre que la goroutine d'affichage ait fini de s'imprimer.
	var calcWg, proxyWg, displayWg sync.WaitGroup

	progressChan := make(chan ProgressUpdate, len(calculators)*10)
	displayWg.Add(1)
	go displayAggregateProgress(&displayWg, progressChan, len(calculators), out)

	for i, calc := range calculators {
		calcWg.Add(1)
		go func(idx int, calculator Calculator) {
			defer calcWg.Done()

			proxyChan := make(chan float64, 100)
			proxyWg.Add(1)
			go func() {
				defer proxyWg.Done()
				for p := range proxyChan {
					progressChan <- ProgressUpdate{CalculatorIndex: idx, Value: p}
				}
			}()

			startTime := time.Now()
			res, err := calculator.Calculate(ctx, proxyChan, config.N)
			close(proxyChan) // Ferme le proxy, ce qui terminera la goroutine du proxy.

			results[idx] = CalculationResult{
				Name:     calculator.Name(),
				Result:   res,
				Duration: time.Since(startTime),
				Err:      err,
			}
		}(i, calc)
	}

	// Le séquence d'attente est cruciale :
	// 1. Attendre la fin des calculs.
	calcWg.Wait()
	// 2. Attendre que toutes les goroutines de proxy aient fini d'envoyer leurs messages.
	proxyWg.Wait()
	// 3. Maintenant, il est sûr de fermer le canal de progression principal.
	close(progressChan)
	// 4. Attendre que la goroutine d'affichage se termine.
	displayWg.Wait()

	return results
}

// analyzeComparisonResults traite les résultats du mode "all" (comparaison).
func analyzeComparisonResults(results []CalculationResult, config AppConfig, out io.Writer) int {
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

	// Vérification cruciale : les résultats sont-ils identiques ?
	for i := 1; i < len(results); i++ {
		if results[i].Result == nil || results[i].Result.Cmp(firstResult) != 0 {
			fmt.Fprintln(out, "\nStatut : Échec critique. Les algorithmes ont produit des résultats différents !")
			return ExitErrorMismatch
		}
	}
	fmt.Fprintln(out, "\nValidation : Tous les résultats sont identiques.")

	displayResult(firstResult, config.N, 0, config.Verbose, out)
	return ExitSuccess
}

// handleCalculationError interprète le type d'erreur reçu et retourne le code
// de sortie approprié, tout en affichant un message clair à l'utilisateur.
func handleCalculationError(err error, duration time.Duration, timeout time.Duration, out io.Writer) int {
	msg := ""
	if duration > 0 {
		msg = fmt.Sprintf(" après %s", duration)
	}
	// `errors.Is` est la manière moderne de vérifier les types d'erreurs "enveloppées".
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

// displayResult affiche le résultat final du calcul F(n) de manière lisible.
func displayResult(result *big.Int, n uint64, duration time.Duration, verbose bool, out io.Writer) {
	fmt.Fprintln(out, "\n--- Données du résultat ---")
	if duration > 0 {
		fmt.Fprintf(out, "Durée d'exécution : %s\n", duration)
	}
	fmt.Fprintf(out, "Taille du résultat : %d bits.\n", result.BitLen())
	resultStr := result.String()
	numDigits := len(resultStr)
	fmt.Fprintf(out, "Nombre de chiffres décimaux : %d\n", numDigits)

	const truncationLimit = 50 // Seuil au-delà duquel on tronque le nombre.
	const displayEdges = 20    // Nombre de chiffres à afficher au début et à la fin.

	if verbose {
		// Le mode verbeux affiche le nombre en entier.
		fmt.Fprintf(out, "F(%d) = %s\n", n, resultStr)
	} else if numDigits > truncationLimit {
		// Par défaut, pour les grands nombres, on affiche le début et la fin pour
		// donner une idée de la magnitude sans polluer le terminal.
		fmt.Fprintf(out, "F(%d) (tronqué) = %s...%s\n", n, resultStr[:displayEdges], resultStr[numDigits-displayEdges:])
	} else {
		// Les nombres "courts" sont affichés en entier.
		fmt.Fprintf(out, "F(%d) = %s\n", n, resultStr)
	}
}

// displayAggregateProgress agrège la progression d'un ou plusieurs calculateurs
// et affiche une barre de progression moyenne.
func displayAggregateProgress(wg *sync.WaitGroup, progressChan <-chan ProgressUpdate, numCalculators int, out io.Writer) {
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

		// Le libellé change si c'est un ou plusieurs calculateurs pour plus de clarté.
		label := "Progression"
		if numCalculators > 1 {
			label = "Progression Globale"
		}
		fmt.Fprintf(out, "\r%s : %6.2f%% [%-30s]", label, avgProgress*100, progressBar(avgProgress, 30))
	}

	for {
		select {
		case update, ok := <-progressChan:
			if !ok {
				// Le canal est fermé, tous les calculs sont terminés.
				for i := range progresses {
					progresses[i] = 1.0 // Force la progression de tous à 100%.
				}
				printBar()
				fmt.Fprintln(out)
				return
			}
			// Mise à jour de la progression pour le calculateur concerné.
			if update.CalculatorIndex < len(progresses) {
				progresses[update.CalculatorIndex] = update.Value
			}
		case <-ticker.C:
			printBar()
		}
	}
}

// progressBar génère la chaîne de caractères représentant la barre de progression.
func progressBar(progress float64, length int) string {
	// Clamp progress to [0, 1]
	if progress > 1.0 {
		progress = 1.0
	} else if progress < 0.0 {
		progress = 0.0
	}
	count := int(progress * float64(length))
	var builder strings.Builder
	builder.Grow(length) // Pré-alloue la mémoire pour l'efficacité.
	for i := 0; i < length; i++ {
		if i < count {
			builder.WriteRune('■') // Caractère pour la partie remplie.
		} else {
			builder.WriteRune(' ') // Caractère pour la partie vide.
		}
	}
	return builder.String()
}
