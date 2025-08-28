import unittest
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, patch

from llm.agents.music_agent import MusicGenerationAgent
from llm.music_gen_service.llm_schemas import Bar, MelodyData, Note


class TestMelodyCorrection(unittest.TestCase):
    def setUp(self):
        # Mock the ChatSession
        self.mock_chat_session = MagicMock()
        
        # Create our agent with the mock
        self.agent = MusicGenerationAgent(chat_session=self.mock_chat_session)
        
        # Mock the get_scale_pitch_classes function to return a specific set of allowed pitches
        # For C major: C(0), D(2), E(4), F(5), G(7), A(9), B(11)
        self.pitch_classes_patcher = patch('pydantic_ai_wrapper.agents.music_agent.get_scale_pitch_classes', 
                                          return_value={0, 2, 4, 5, 7, 9, 11})
        self.mock_get_scale_pitch_classes = self.pitch_classes_patcher.start()
        
        # Sample chord analysis data for tests
        self.mock_chord_analysis = [
            {
                "chord_name": "C",
                "note_weights": {
                    "C": 100.0,  # Root - highest weight
                    "E": 50.0,   # Third
                    "G": 75.0,   # Fifth
                    "D": 10.0,   # Non-chord tone
                    "F": 15.0,   # Non-chord tone
                    "A": 5.0,    # Non-chord tone
                    "B": 2.0     # Non-chord tone
                }
            }
        ]

    def tearDown(self):
        self.pitch_classes_patcher.stop()

    def test_closest_note_priority(self):
        """Test that the closest in-key note is selected over notes with higher harmonic weight."""
        # Create a melody with an out-of-key note (C#/1)
        melody_data = MelodyData(bars=[
            Bar(bar=1, notes=[
                Note(pitch=61, start_beat=0.0, duration_beats=1.0)  # C# (out of key)
            ])
        ])
        
        # In C major, both C(60/0) and D(62/2) are equally distant from C#(61) but 
        # C has higher harmonic weight as the root of the chord
        corrected_melody = self.agent._correct_notes_in_key(
            melody_data=melody_data,
            key_name="C",
            mode_name="major",
            chord_analysis_data=self.mock_chord_analysis,
            chord_progression_str="C",
            duration_bars=1
        )
        
        # Verify C(60) was chosen as it has higher harmonic weight (100.0 vs 10.0 for D)
        # when both are equidistant (1 semitone away)
        self.assertEqual(corrected_melody.bars[0].notes[0].pitch, 60)
    
    def test_harmonic_weight_tiebreaker(self):
        """Test that when two notes are equally distant, the one with higher harmonic weight is chosen."""
        # Create a melody with an out-of-key note (D#/3) 
        # D#(3) is equally distant from D(2) and E(4)
        melody_data = MelodyData(bars=[
            Bar(bar=1, notes=[
                Note(pitch=63, start_beat=0.0, duration_beats=1.0)  # D# (out of key)
            ])
        ])
        
        # In our mock analysis, E(4) would have higher harmonic weight (50.0) than D(2) (10.0)
        corrected_melody = self.agent._correct_notes_in_key(
            melody_data=melody_data,
            key_name="C",
            mode_name="major",
            chord_analysis_data=self.mock_chord_analysis,
            chord_progression_str="C",
            duration_bars=1
        )
        
        # Verify E(64) was chosen due to higher harmonic weight in the tiebreaker
        self.assertEqual(corrected_melody.bars[0].notes[0].pitch, 64)
    
    def test_multiple_notes_correction(self):
        """Test correcting multiple out-of-key notes in a melody."""
        # Create a melody with multiple out-of-key notes
        melody_data = MelodyData(bars=[
            Bar(bar=1, notes=[
                Note(pitch=61, start_beat=0.0, duration_beats=1.0),  # C# (out of key)
                Note(pitch=63, start_beat=1.0, duration_beats=1.0),  # D# (out of key)
                Note(pitch=66, start_beat=2.0, duration_beats=1.0),  # F# (out of key)
                Note(pitch=68, start_beat=3.0, duration_beats=1.0)   # G# (out of key)
            ])
        ])
        
        corrected_melody = self.agent._correct_notes_in_key(
            melody_data=melody_data,
            key_name="C",
            mode_name="major",
            chord_analysis_data=self.mock_chord_analysis,
            chord_progression_str="C",
            duration_bars=1
        )
        
        # Expected corrections based on harmonic weight and voice leading:
        # C#(61) -> C(60) (Harmonic weight 100.0 vs 10.0 for D)
        # D#(63) -> E(64) (Harmonic weight 50.0 vs 10.0 for D)
        # F#(66) -> G(67) (Harmonic weight 75.0 higher than other options)
        # G#(68) -> G(67) (Descending, harmonic weight 75.0 higher than 5.0 for A)
        expected_pitches = [60, 64, 67, 67]
        actual_pitches = [note.pitch for note in corrected_melody.bars[0].notes]
        
        self.assertEqual(actual_pitches, expected_pitches)


if __name__ == "__main__":
    unittest.main() 