package xxhash

import "errors"

// ErrInvalidCheckpoint is returned when a streaming checkpoint cannot be
// decoded: the input is empty, truncated, or oversized; it carries an
// unknown version; its checksum does not match; or it encodes a state
// outside the valid range (for example a tail length of 32 bytes or more,
// or a tail length inconsistent with the total length). Decoding never
// modifies the receiver when this error is returned.
var ErrInvalidCheckpoint = errors.New("xxhash: invalid checkpoint")

// Checkpoint layout (all integers little-endian):
//
//	magic    4 bytes: "xxc" + format version byte
//	seed     8 bytes: seed the Digest was last reset with
//	total    8 bytes: total bytes written so far
//	v1..v4   32 bytes: internal lanes
//	n        1 byte:  number of tail bytes buffered in mem (0..31)
//	mem      32 bytes: tail bytes, zero-padded
//	checksum 8 bytes: Sum64 of all preceding bytes
const (
	checkpointMagic    = "xxc\x01" // "xxc" + format version 1
	checkpointHeadSize = len(checkpointMagic) + 8*6 + 1 + 32
	checkpointSize     = checkpointHeadSize + 8
	checkpointMaxTail  = 32 // tail must stay within one block
)

// Checkpoint returns a verifiable snapshot of d's current streaming state:
// the seed, the total number of bytes written, the internal lanes, and the
// buffered tail bytes. The snapshot is a self-contained copy; d may keep
// accepting writes without affecting it, and snapshots may be resumed
// independently of d.
//
// Checkpoint only reads d; like Sum64 it must not run concurrently with
// Write on the same Digest.
func (d *Digest) Checkpoint() []byte {
	b := make([]byte, 0, checkpointSize)
	b = append(b, checkpointMagic...)
	b = appendUint64(b, d.seed)
	b = appendUint64(b, d.total)
	b = appendUint64(b, d.v1)
	b = appendUint64(b, d.v2)
	b = appendUint64(b, d.v3)
	b = appendUint64(b, d.v4)
	b = append(b, byte(d.n))
	var mem [checkpointMaxTail]byte
	copy(mem[:], d.mem[:d.n])
	b = append(b, mem[:]...)
	b = appendUint64(b, Sum64(b))
	return b
}

// NewFromCheckpoint creates a new Digest resumed from a checkpoint produced
// by Checkpoint. The result is a fork: it shares no state with the Digest
// the checkpoint was taken from, and writing the same subsequent bytes to
// both yields the same Sum64.
func NewFromCheckpoint(b []byte) (*Digest, error) {
	d := new(Digest)
	if err := d.UnmarshalCheckpoint(b); err != nil {
		return nil, err
	}
	return d, nil
}

// UnmarshalCheckpoint restores d from a checkpoint produced by Checkpoint.
// The checkpoint is fully validated — size, version, checksum, and state
// ranges — before d is touched; on error d is left unmodified and
// ErrInvalidCheckpoint is returned. Decoding allocates nothing beyond the
// size of the input.
func (d *Digest) UnmarshalCheckpoint(b []byte) error {
	if len(b) != checkpointSize {
		return ErrInvalidCheckpoint
	}
	if !hasCheckpointMagic(b) {
		return ErrInvalidCheckpoint
	}
	if u64(b[checkpointHeadSize:]) != Sum64(b[:checkpointHeadSize]) {
		return ErrInvalidCheckpoint
	}

	b = b[len(checkpointMagic):]
	b, seed := consumeUint64(b)
	b, total := consumeUint64(b)
	b, v1 := consumeUint64(b)
	b, v2 := consumeUint64(b)
	b, v3 := consumeUint64(b)
	b, v4 := consumeUint64(b)
	n := int(b[0])
	b = b[1:]
	if n >= checkpointMaxTail {
		return ErrInvalidCheckpoint
	}
	if uint64(n) != total%uint64(checkpointMaxTail) {
		return ErrInvalidCheckpoint
	}

	// Everything validated; only now modify d.
	d.seed = seed
	d.total = total
	d.v1, d.v2, d.v3, d.v4 = v1, v2, v3, v4
	d.n = n
	copy(d.mem[:], b[:checkpointMaxTail])
	return nil
}

func hasCheckpointMagic(b []byte) bool {
	for i := 0; i < len(checkpointMagic); i++ {
		if b[i] != checkpointMagic[i] {
			return false
		}
	}
	return true
}
