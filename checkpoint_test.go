package xxhash

import (
	"bytes"
	"testing"
)

// testPattern returns a deterministic pseudo-random byte slice of length n.
func testPattern(n int) []byte {
	b := make([]byte, n)
	x := uint64(0x9e3779b97f4a7c15)
	for i := range b {
		x ^= x << 13
		x ^= x >> 7
		x ^= x << 17
		b[i] = byte(x)
	}
	return b
}

func TestCheckpointFork(t *testing.T) {
	data := testPattern(1000)
	for _, split := range []int{0, 1, 7, 31, 32, 33, 63, 64, 100, 999, 1000} {
		for _, seed := range []uint64{0, 42} {
			d := NewWithSeed(seed)
			d.Write(data[:split])
			cp := d.Checkpoint()

			fork, err := NewFromCheckpoint(cp)
			if err != nil {
				t.Fatalf("split=%d seed=%d: NewFromCheckpoint: %v", split, seed, err)
			}
			// Both instances continue with the same bytes.
			d.Write(data[split:])
			fork.Write(data[split:])
			if got, want := fork.Sum64(), d.Sum64(); got != want {
				t.Errorf("split=%d seed=%d: fork Sum64 = %#x, want %#x", split, seed, got, want)
			}
			// And the fork matches a one-shot digest of the full input.
			full := NewWithSeed(seed)
			full.Write(data)
			if got, want := fork.Sum64(), full.Sum64(); got != want {
				t.Errorf("split=%d seed=%d: fork Sum64 = %#x, want one-shot %#x", split, seed, got, want)
			}
		}
	}
}

func TestCheckpointDoesNotDisturbOriginal(t *testing.T) {
	data := testPattern(300)
	d := New()
	d.Write(data[:50])
	want := New()
	want.Write(data[:50])

	cp1 := d.Checkpoint()
	d.Write(data[50:])
	cp2 := d.Checkpoint()

	want.Write(data[50:])
	if got := d.Sum64(); got != want.Sum64() {
		t.Errorf("Sum64 after Checkpoint calls = %#x, want %#x", got, want.Sum64())
	}
	if bytes.Equal(cp1, cp2) {
		t.Error("checkpoints at different positions are identical")
	}

	// Resuming the first checkpoint replays from the midpoint.
	fork, err := NewFromCheckpoint(cp1)
	if err != nil {
		t.Fatal(err)
	}
	fork.Write(data[50:])
	if got := fork.Sum64(); got != want.Sum64() {
		t.Errorf("resumed checkpoint Sum64 = %#x, want %#x", got, want.Sum64())
	}
}

func TestCheckpointRecordsSeed(t *testing.T) {
	const seed = 0xdeadbeef
	d := NewWithSeed(seed)
	d.Write(testPattern(10))
	fork, err := NewFromCheckpoint(d.Checkpoint())
	if err != nil {
		t.Fatal(err)
	}
	if fork.seed != seed {
		t.Errorf("fork seed = %d, want %d", fork.seed, seed)
	}
}

// tamperCheckpoint returns a copy of cp with the mutation fn applied and the
// checksum recomputed, so the result passes the checksum check and exercises
// the semantic validations.
func tamperCheckpoint(cp []byte, fn func(b []byte)) []byte {
	b := append([]byte(nil), cp...)
	fn(b)
	b = b[:checkpointHeadSize]
	b = appendUint64(b, Sum64(b))
	return b
}

func TestUnmarshalCheckpointErrors(t *testing.T) {
	d := NewWithSeed(7)
	d.Write(testPattern(77))
	valid := d.Checkpoint()

	cases := map[string][]byte{
		"nil":         nil,
		"empty":       {},
		"truncated":   valid[:checkpointSize-1],
		"truncated10": valid[:10],
		"oversized":   append(append([]byte(nil), valid...), 0),
		"bad version": tamperCheckpoint(valid, func(b []byte) { b[3] = 0x02 }),
		"bad magic":   tamperCheckpoint(valid, func(b []byte) { b[0] = 'X' }),
		"tail too long": tamperCheckpoint(valid, func(b []byte) {
			b[len(checkpointMagic)+8*6] = 32
		}),
		"tail way too long": tamperCheckpoint(valid, func(b []byte) {
			b[len(checkpointMagic)+8*6] = 200
		}),
		"length mismatch": tamperCheckpoint(valid, func(b []byte) {
			// Bump total without adjusting the tail length.
			b[len(checkpointMagic)+8] ^= 0x01
		}),
	}
	// A flipped payload byte with the original (now stale) checksum.
	flipped := append([]byte(nil), valid...)
	flipped[20] ^= 0xff
	cases["checksum mismatch"] = flipped

	for name, cp := range cases {
		t.Run(name, func(t *testing.T) {
			fresh := New()
			fresh.Write(testPattern(5))
			before := fresh.Sum64()
			if err := fresh.UnmarshalCheckpoint(cp); err != ErrInvalidCheckpoint {
				t.Fatalf("UnmarshalCheckpoint error = %v, want ErrInvalidCheckpoint", err)
			}
			if got := fresh.Sum64(); got != before {
				t.Errorf("failed decode modified the digest: Sum64 = %#x, want %#x", got, before)
			}
			if _, err := NewFromCheckpoint(cp); err != ErrInvalidCheckpoint {
				t.Errorf("NewFromCheckpoint error = %v, want ErrInvalidCheckpoint", err)
			}
		})
	}
}

func TestCheckpointRoundTripThroughWrite(t *testing.T) {
	// Simulate an upload resumed from a checkpoint: write in chunks,
	// checkpoint mid-stream, then continue on a fresh instance.
	data := testPattern(4096)
	d := New()
	var cp []byte
	for i := 0; i < len(data); i += 100 {
		end := i + 100
		if end > len(data) {
			end = len(data)
		}
		d.Write(data[i:end])
		if i == 2000 {
			cp = d.Checkpoint()
		}
	}
	want := d.Sum64()

	resumed, err := NewFromCheckpoint(cp)
	if err != nil {
		t.Fatal(err)
	}
	resumed.Write(data[2100:])
	if got := resumed.Sum64(); got != want {
		t.Errorf("resumed Sum64 = %#x, want %#x", got, want)
	}
}

func TestUnmarshalCheckpointNoAllocs(t *testing.T) {
	d := New()
	d.Write(testPattern(40))
	cp := d.Checkpoint()
	allocs := testing.AllocsPerRun(100, func() {
		if err := d.UnmarshalCheckpoint(cp); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Errorf("UnmarshalCheckpoint allocated %v times, want 0", allocs)
	}
}

func TestCheckpointAfterUnmarshalBinary(t *testing.T) {
	// A digest restored from the legacy binary format can still be
	// checkpointed and resumed; the seed is recorded as zero.
	d := NewWithSeed(9)
	d.Write(testPattern(50))
	state, err := d.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	var legacy Digest
	if err := legacy.UnmarshalBinary(state); err != nil {
		t.Fatal(err)
	}
	fork, err := NewFromCheckpoint(legacy.Checkpoint())
	if err != nil {
		t.Fatal(err)
	}
	more := testPattern(20)
	d.Write(more)
	fork.Write(more)
	if got, want := fork.Sum64(), d.Sum64(); got != want {
		t.Errorf("fork of legacy-restored digest = %#x, want %#x", got, want)
	}
}
