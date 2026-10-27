// Package split assigns corpus samples to train or test deterministically.
package split

import "hash/fnv"

// Of returns "test" for ~20% of ids and "train" for the rest. A human text
// and its AI mirror share an id, so pairs never straddle the split.
func Of(id string) string {
	h := fnv.New32a()
	h.Write([]byte(id))
	if h.Sum32()%5 == 0 {
		return "test"
	}
	return "train"
}
