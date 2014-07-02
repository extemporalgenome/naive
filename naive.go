package naive

import (
	"math"
)

var defaultProb = math.Log(0.00000000001)

// Clasifier is an implementation of a naive Bayes classifier.
type Classifier struct {
	// Total word count.
	total float64

	// Total word count, by class.
	totals []float64

	// Training data, by word.
	data map[string][]cell
}

// Holds for a training data for a word+class pair.
type cell struct {
	class int
	count float64
}

// NewClassifier initializes a new Classifier.
func NewClassifier() *Classifier {
	return &Classifier{0, nil, make(map[string][]cell)}
}

// Train adds training data to the classifier.
func (c *Classifier) Train(doc []string, class int) {
	if class < 0 {
		panic("naive: class < 0")
	}

	// Grow the totals slice if need be.
	if class >= len(c.totals) {
		grown := make([]float64, class+1)
		copy(grown, c.totals)
		c.totals = grown
	}

	n := float64(len(doc))
	c.totals[class] += n
	c.total += n

	for _, word := range doc {
		c.data[word] = incr(c.data[word], class)
	}
}

// incr either finds an already allocated cell for the class and increments its
// count, or inserts a new cell, with a count of 1, in the correct position.
func incr(s []cell, c int) []cell {
	// TODO: Perform a binary search here instead?
	for i := range s {
		if cur := &s[i]; cur.class == c {
			cur.count++
			return s
		} else if cur.class > c {
			s = append(s, cell{})
			copy(s[i+1:], s[i:])
			s[i] = cell{c, 1}
			return s
		}
	}

	return append(s, cell{c, 1})
}

// Untrain undoes a previous classification
func (c *Classifier) Untrain(doc []string, class int) {
	if class < 0 {
		panic("naive: class < 0")
	}

	// Warn about classes we haven't seen before.
	if class >= len(c.totals) {
		panic("naive: unknown class")
	}

	n := float64(len(doc))
	c.totals[class] -= n
	c.total -= n

	for _, word := range doc {
		cells := decr(c.data[word], class)
		if len(cells) == 0 {
			delete(c.data, word)
		} else {
			c.data[word] = cells
		}
	}
}

// decr does the exact opposite of incr.
func decr(s []cell, c int) []cell {
	// TODO: Perform a binary search here instead?
	for i := range s {
		if cur := &s[i]; cur.class == c {
			if cur.count > 1 {
				cur.count--
			} else {
				return append(s[:i], s[i+1:]...)
			}
		} else if cur.class > c {
			break
		}
	}

	return s
}

// Classify classifies a document.
func (c *Classifier) Classify(doc []string) (class int, tied bool, scores []float64) {
	scores = make([]float64, len(c.totals))

	// For every word in the document, calculate each class' probability.
	for _, word := range doc {
		i := 0

		for _, cur := range c.data[word] {
			// Use the default score for missing classes.
			for i < cur.class {
				scores[i] += defaultProb
				i++
			}

			scores[i] += math.Log(cur.count / c.totals[i])
			i++
		}

		// Again, use the default score for missing classes.
		for i < len(scores) {
			scores[i] += defaultProb
			i++
		}
	}

	var max float64

	// Add each class' prior probability to its score.
	for i, score := range scores {
		score += math.Log(c.totals[i] / c.total)

		// While we're iterating over the scores slice anyway, look for the
		// best candidate.
		if score > max {
			class = i
			max = score
			tied = false
		} else if score == max {
			tied = true
		}

		scores[i] = score
	}

	return class, tied, scores
}
