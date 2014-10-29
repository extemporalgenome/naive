package naive

import "testing"

type doc struct {
	words []string
	class int
}

type check struct {
	words   []string
	classes []int
	tied    bool
}

type test struct {
	docs   []doc
	checks []check
}

var tests = []test{
	{
		docs: []doc{
			{
				words: []string{"alpha", "beta"},
				class: 0,
			}, {
				words: []string{"beta", "gamma"},
				class: 1,
			}, {
				words: []string{"gamma", "delta"},
				class: 2,
			},
		},
		checks: []check{
			{
				words:   []string{"alpha"},
				classes: []int{0},
				tied:    false,
			}, {
				words:   []string{"alpha", "beta"},
				classes: []int{0},
				tied:    false,
			}, {
				words:   []string{"beta"},
				classes: []int{0, 1},
				tied:    true,
			}, {
				words:   []string{"beta", "gamma"},
				classes: []int{1},
				tied:    false,
			},
		},
	},
}

func TestNaive(t *testing.T) {
	c := NewClassifier()
	for i, test := range tests {
		for _, doc := range test.docs {
			t.Logf("Train(%v, %d)", doc.words, doc.class)
			c.Train(doc.words, doc.class)
		}
		for j, check := range test.checks {
			class, tied, scores := c.Classify(check.words)
			ok := false
			// ensure that check.classes contains the returned class
			for _, candidate := range check.classes {
				if class == candidate {
					ok = true
					break
				}
			}
			switch {
			case !ok:
				t.Errorf("Classify(%v) = %d (class), want value in %v; test %d, check %d",
					check.words, class, check.classes, i, j)
			case tied != check.tied:
				t.Errorf("Classify(%v) = %t (tied), want value in %t; test %d, check %d",
					check.words, tied, check.tied, i, j)
			}
			t.Logf("Classify(%v) = %v (scores); test %d, check %d",
				check.words, scores, i, j)
		}
	}
}
