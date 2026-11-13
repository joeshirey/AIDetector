package engine

import (
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"runtime/debug"
	"slices"
	"sync"
)

const shortCommitLen = 12

// BuildInfo identifies the binary build and loaded model set that produced a result.
type BuildInfo struct {
	Version   string `json:"version" jsonschema:"module version, or devel/unknown for untagged builds"`
	Commit    string `json:"commit,omitempty" jsonschema:"short git commit from VCS build info, with -dirty if modified"`
	GoVersion string `json:"go_version,omitempty" jsonschema:"Go toolchain version"`
	Models    string `json:"models,omitempty" jsonschema:"sha256 digest over the loaded model files"`
}

type loadedFile struct {
	name string
	data []byte
}

var cachedBuildInfo = sync.OnceValue(func() BuildInfo {
	return parseBuildInfo(debug.ReadBuildInfo())
})

// BuildInfo returns the build version and loaded-models identifier for this engine.
func (e *Engine) BuildInfo() BuildInfo {
	bi := cachedBuildInfo()
	if e != nil {
		bi.Models = e.modelsDigest
	}
	return bi
}

func parseBuildInfo(bi *debug.BuildInfo, ok bool) BuildInfo {
	if !ok || bi == nil {
		return BuildInfo{Version: "unknown"}
	}
	ver := bi.Main.Version
	switch ver {
	case "":
		ver = "unknown"
	case "(devel)":
		ver = "devel"
	}
	var rev string
	var modified bool
	for _, s := range bi.Settings {
		switch s.Key {
		case "vcs.revision":
			rev = s.Value
		case "vcs.modified":
			modified = s.Value == "true"
		}
	}
	if len(rev) > shortCommitLen {
		rev = rev[:shortCommitLen]
	}
	if rev != "" && modified {
		rev += "-dirty"
	}
	return BuildInfo{
		Version:   ver,
		Commit:    rev,
		GoVersion: bi.GoVersion,
	}
}

func hashLoadedModels(files []loadedFile) string {
	if len(files) == 0 {
		return ""
	}
	sorted := slices.Clone(files)
	slices.SortFunc(sorted, func(a, b loadedFile) int {
		return cmp.Compare(a.name, b.name)
	})
	h := sha256.New()
	for _, f := range sorted {
		fmt.Fprintf(h, "%d:%s\n%d:", len(f.name), f.name, len(f.data))
		h.Write(f.data)
		h.Write([]byte{'\n'})
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}
