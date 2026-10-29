// Package split assigns corpus samples to train or test deterministically.
package split

import (
	"hash/fnv"
	"strings"
)

// Of returns "test" for ~20% of ids and "train" for the rest. A human text
// and its AI mirror share an id, so pairs never straddle the split.
// Variant samples are named "<id>~<style>" (another generation from the
// same human source); they belong to their source's split.
func Of(id string) string {
	if Hash(id)%5 == 0 {
		return "test"
	}
	return "train"
}

// Hash is the stable id hash behind the split. Seed other assignments (like
// cross-validation folds) from Hash(id)/5 so they're independent of the split.
func Hash(id string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(Base(id)))
	return h.Sum32()
}

// Base strips a "~variant" suffix from a sample id.
func Base(id string) string {
	if i := strings.IndexByte(id, '~'); i >= 0 {
		return id[:i]
	}
	return id
}
