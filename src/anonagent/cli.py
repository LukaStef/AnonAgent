"""Console demo.

Two modes. Without --ask the whole thing runs offline, so the privacy layer
can be shown and debugged before any API key exists. With --ask the masked
question actually goes to a cloud model and the answer is restored locally.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from anonagent.agent.llm import build_llm
from anonagent.agent.pipeline import LeakDetected, ModelUnavailable, PrivacyPipeline
from anonagent.demo_data import DEFAULT_SAMPLE, SAMPLES
from anonagent.documents import DEFAULT_CHUNK_CHARS
from anonagent.privacy import Masker, PiiDetector, PseudonymVault
from anonagent.review import InteractiveReviewer

BAR = "=" * 72


def main() -> int:
    load_dotenv()
    args = _parse_args()
    text = _read_input(args)

    masker = Masker(
        detector=PiiDetector(
            detect_money=args.detect_money,
            score_threshold=args.score_threshold,
            mask_malformed=args.include_malformed,
        ),
        vault=PseudonymVault(secret=args.secret, match=args.match),
        reviewer=InteractiveReviewer(args.confirm_below) if args.review else None,
    )

    try:
        if args.file:
            return _mask_file(masker, text, args)
        if args.chat:
            pipeline = PrivacyPipeline(build_llm(args.provider, args.model), masker)
            return _chat(pipeline)
        return _ask(masker, text, args) if args.ask else _show_masking(masker, text)
    except LeakDetected as error:
        print(f"\nBLOCKED: {error}", file=sys.stderr)
        return 2
    except ModelUnavailable as error:
        # Nothing private was at risk here: the masked text is all that was
        # ever sent, and the failure happened after masking.
        print(f"\nCould not reach the model: {error}", file=sys.stderr)
        return 3
    except OSError as error:
        print(f"\nCannot read or write that file: {error}", file=sys.stderr)
        return 1
    except RuntimeError as error:
        print(f"\nError: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # noqa: RUF100
        print("\nStopped.", file=sys.stderr)
        return 130


def _chat(pipeline: PrivacyPipeline, stream: object | None = None) -> int:
    """Ask follow-up questions in one session, sharing one vault and thread.

    Each turn prints what actually left the machine, which is the point: the
    provider follows the conversation while never learning a name.
    """
    read = _line_reader(stream)
    print("Chat mode. Blank line or Ctrl-D to leave.\n")

    while True:
        question = read("you> ")
        if question is None or not question.strip() or question.strip() in ("exit", "quit"):
            print("Bye.")
            return 0

        try:
            answer = pipeline.ask(question)
        except LeakDetected as error:
            print(f"  BLOCKED: {error}\n", file=sys.stderr)
            continue
        except ModelUnavailable as error:
            print(f"  Could not reach the model: {error}\n", file=sys.stderr)
            continue

        print(f"  -> cloud sees: {answer.masked_question}")
        print(f"  <- {answer.answer}\n")


def _line_reader(stream: object | None):
    """Read a line, returning None at end of input rather than raising."""

    def read(prompt: str) -> str | None:
        if stream is not None:
            line = stream.readline()
            return None if line == "" else line.rstrip("\n")
        try:
            return input(prompt)
        except EOFError:
            return None

    return read


def _mask_file(masker: Masker, text: str, args: argparse.Namespace) -> int:
    """Sanitize a whole document and write the safe version out."""
    source = Path(args.file)
    result = masker.mask_document(text, max_chars=args.chunk_size)

    destination = Path(args.out) if args.out else source.with_suffix(f".masked{source.suffix}")
    if destination.exists() and not args.overwrite:
        print(f"\nRefusing to overwrite {destination}. Pass --overwrite.", file=sys.stderr)
        return 1
    destination.write_text(result.masked_text, encoding="utf-8")

    counts = result.entity_counts
    print(f"{source} -> {destination}")
    print(f"  {len(text):,} chars, {sum(counts.values())} values masked")
    for entity_type, count in sorted(counts.items()):
        print(f"    {entity_type:<18} x{count}")
    if result.sealed:
        print(f"  + sweep sealed {len(result.sealed)} mention(s) the detector missed")
    for entity in result.declined:
        print(f"  ! LEFT IN THE CLEAR by your choice: {entity.entity_type} {entity.text!r}")

    leaked = masker.leaks(result.masked_text)
    if leaked:
        print(f"\n{len(leaked)} private value(s) survived masking. Not safe to share.", file=sys.stderr)
        return 2
    return 0


def _show_masking(masker: Masker, text: str) -> int:
    """Offline: mask, then restore, with no model involved."""
    # Print the input before masking: with --review the user is about to be
    # asked about values in it, and being asked about text you have not seen
    # yet is no way to make a decision.
    _panel("1. LOCAL INPUT (never leaves this machine)", text)

    result = masker.mask(text)
    _detections(result)
    _panel(_outbound_title(result.declined), result.masked_text)

    restored = masker.unmask(result.masked_text)
    _panel("4. RESTORED LOCALLY", restored)
    return _report_round_trip(masker, result, restored)


def _ask(masker: Masker, text: str, args: argparse.Namespace) -> int:
    """Online: the full round trip through a real model."""
    llm = build_llm(args.provider, args.model)
    pipeline = PrivacyPipeline(llm, masker)

    _panel("1. LOCAL INPUT (never leaves this machine)", text)
    answer = pipeline.ask(text)

    _counts(answer)
    _panel("3. SENT TO CLOUD", answer.masked_question)
    _panel("4. CLOUD MODEL REPLIED (still masked)", answer.masked_answer)
    _panel("5. RESTORED LOCALLY", answer.answer)

    print(f"\n{BAR}\nAGENT LOG\n{BAR}")
    for event in answer.events:
        marker = "LOCAL " if event.sensitive else "public"
        print(f"  [{marker}] {event.step:<16} {event.message}")
    return 0


def _counts(answer) -> None:
    """In ask mode the spans are already masked, so report counts, not values."""
    print(f"\n{BAR}\n2. DETECTED: {sum(answer.entity_counts.values())} sensitive spans\n{BAR}")
    for entity_type, count in sorted(answer.entity_counts.items()):
        print(f"  {entity_type:<18} x{count}")
    if not answer.entity_counts:
        print("  (nothing detected)")


def _detections(result) -> None:
    if result is None:
        return
    print(f"\n{BAR}\n2. DETECTED: {len(result.entities)} sensitive spans\n{BAR}")
    for entity in result.entities:
        print(f"  {entity.entity_type:<18} score={entity.score:.2f}  {entity.text!r}")
    if not result.entities:
        print("  (nothing detected)")
    if result.sealed:
        print(f"  + sweep sealed {len(result.sealed)} later mention(s) the detector missed")
    for entity in result.declined:
        print(f"  ! LEFT IN THE CLEAR by your choice: {entity.entity_type} {entity.text!r}")


def _outbound_title(declined) -> str:
    """Never call the payload safe when the user chose to expose part of it."""
    if not declined:
        return "3. SENT TO CLOUD (safe)"
    return f"3. SENT TO CLOUD ({len(declined)} value(s) exposed by your choice)"


def _panel(title: str, body: str) -> None:
    print(f"\n{BAR}\n{title}\n{BAR}\n{body}")


def _report_round_trip(masker: Masker, result, restored: str) -> int:
    """Say exactly what came back, and why it differs when it does."""
    if restored == result.original_text:
        print("\nRound trip: byte-identical.")
        return 0

    canonicalized = masker.vault.canonicalized()
    if not canonicalized:
        print("\nRound trip FAILED: text changed for reasons other than canonicalization.")
        return 1

    print("\nRound trip: all values restored, some in canonical form.")
    for pseudonym in canonicalized:
        spellings = masker.vault.variants(pseudonym)
        print(f"  {pseudonym.placeholder} saw {spellings!r} -> all restored as {spellings[0]!r}")
    print("  Use --match exact for a byte-identical round trip.")
    return 0


def _read_input(args: argparse.Namespace) -> str:
    """Text from the argument, from stdin on an explicit "-", or a sample.

    Reading stdin only on "-" is deliberate. Deciding by isatty() looks
    friendlier but hangs forever whenever stdin is an open pipe with nothing
    in it, which is every CI job and every script that redirects.
    """
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    if args.text == "-":
        return sys.stdin.read().strip()
    if args.text:
        return args.text
    return SAMPLES[args.sample]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mask sensitive text locally, optionally ask a cloud model, restore locally.",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help='Text to use. "-" reads stdin. Defaults to a built-in sample.',
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Read a document from this path, write a masked copy, and stop. "
        "Long documents are analyzed in chunks that share one vault.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Where the masked copy goes. Defaults to <name>.masked<ext>.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow --out to replace an existing file.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_CHARS,
        help="Characters per chunk when masking a file.",
    )
    parser.add_argument(
        "--ask",
        action="store_true",
        help="Actually send the masked text to a cloud model. Needs an API key.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Ask follow-up questions in one session. The thread that is "
        "replayed to the model is the masked one.",
    )
    parser.add_argument(
        "--sample",
        default=DEFAULT_SAMPLE,
        choices=sorted(SAMPLES),
        help="Which built-in sample to use when no text is given.",
    )
    parser.add_argument("--provider", default=None, help="openai, groq or ollama.")
    parser.add_argument("--model", default=None, help="Model name for the chosen provider.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.4,
        help="Lowest detector confidence still treated as sensitive. Lower masks "
        "more and leaks less; raise it to watch the guard come apart.",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Ask before masking anything the detector is unsure about. "
        "Pairs well with a low --score-threshold, which surfaces more "
        "candidates for you to rule on.",
    )
    parser.add_argument(
        "--confirm-below",
        type=float,
        default=0.5,
        help="With --review, detections below this score are put to you. "
        "Anything at or above it is masked without asking.",
    )
    parser.add_argument(
        "--include-malformed",
        action="store_true",
        help="Also mask identifiers that fail their own checksum. Off by "
        "default, so only account and identity numbers that check out are "
        "masked -- a mistyped one then travels in the clear.",
    )
    parser.add_argument(
        "--detect-money",
        action="store_true",
        help="Also mask currency amounts. Off by default: masking the numbers "
        "is what stops the model from reasoning about them.",
    )
    parser.add_argument(
        "--match",
        default="normalized",
        choices=("normalized", "exact"),
        help="normalized: spellings of one value share a placeholder, and come "
        "back spelled like the first. exact: only identical spellings share a "
        "placeholder, and the round trip is byte-identical.",
    )
    parser.add_argument(
        "--secret",
        default=None,
        help="HMAC secret, for fingerprints that stay stable across runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
