package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/joeshirey/AIDetector/internal/corpus"
)

func runPing(ctx context.Context, gens string) error {
	gs, err := corpus.NewGenerators(ctx, gens)
	if err != nil {
		return err
	}
	for _, g := range gs {
		out, err := g.Generate(ctx, "Reply with exactly: ok")
		if err != nil {
			fmt.Printf("%-26s FAIL %v\n", g.Name(), err)
			continue
		}
		fmt.Printf("%-26s %s\n", g.Name(), strings.TrimSpace(out))
	}
	fmt.Println(corpus.Spend)
	return nil
}
