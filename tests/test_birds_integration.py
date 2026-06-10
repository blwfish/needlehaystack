"""Integration tests for the birds domain.

These tests run the real captioner against real photos via a live Ollama instance.
They are slow (~5-15s per photo) and require network access on first run to download
test fixtures from Wikipedia.

Run with:
    pytest -m integration
    pytest -m integration -v -s   # show print output (species names the model returned)

Skipped automatically if Ollama is not reachable.
"""
import pytest
from PIL import Image

from needlestack_core.captioner import Captioner
from needlestack_core.taxonomy import BIRDS


@pytest.mark.integration
class TestBirdsCaptioner:

    @pytest.fixture(autouse=True)
    def _require_ollama(self, ollama_running):
        pass

    def _caption(self, path):
        captioner = Captioner(domain=BIRDS)
        return captioner.caption(Image.open(path))

    def test_bird_photos_recognized_as_on_topic(self, bird_photos):
        """Every bird photo should return is_railroad=True (the generic 'on-topic' flag)."""
        for group, path in bird_photos.items():
            result = self._caption(path)
            assert result.is_railroad, (
                f"{group} photo ({path.name}) returned is_birds=False — "
                f"caption: {result.caption!r}"
            )

    def test_bird_photos_have_common_name(self, bird_photos):
        """common_name is a high-weight field; it should always be populated."""
        for group, path in bird_photos.items():
            result = self._caption(path)
            assert result.reporting_marks.strip(), (
                f"{group} photo: reporting_marks (common_name/species) is empty — "
                f"structured_json: {result.structured_json}"
            )

    def test_bird_photos_have_equipment_type(self, bird_photos):
        """type and behavior are mid-weight fields; at least one should be populated."""
        for group, path in bird_photos.items():
            result = self._caption(path)
            assert result.equipment.strip(), (
                f"{group} photo: equipment (type/behavior/plumage) is empty — "
                f"structured_json: {result.structured_json}"
            )

    def test_bird_photos_have_caption(self, bird_photos):
        """Synthesized caption text should be non-empty."""
        for group, path in bird_photos.items():
            result = self._caption(path)
            assert result.caption.strip(), f"{group} photo: caption is empty"

    def test_non_bird_returns_false(self, non_bird_photo):
        """A landscape photo should return is_birds=False."""
        result = self._caption(non_bird_photo)
        assert not result.is_railroad, (
            f"Non-bird photo was flagged as on-topic — caption: {result.caption!r}"
        )

    def test_species_identification_soft(self, bird_photos):
        """Print what the model identified — not asserted, but useful for spot-checking."""
        for group, path in bird_photos.items():
            result = self._caption(path)
            print(f"\n  {group}: marks={result.reporting_marks!r}  equip={result.equipment!r}")
