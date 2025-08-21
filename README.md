# Leveraging RAG for a Low-Resource Audio-Aware Diachronic Analysis of Gendered Toy Marketing

Luca Marinelli<sup>1</sup>, Iacopo Ghinassi<sup>2</sup>, Charalampos Saitis<sup>1</sup>

<sup>1</sup>Centre for Digital Music, Queen Mary University of London, UK

<sup>2</sup>College of Computing and Data Science, Nanyang Technological University, Singapore

Accepted at RANLP 2025 at the [Workshop on Natural Language Processing and Language Models for Digital Humanities](https://ranlp.org/ranlp2025/index.php/workshops/)

#### Abstract
> We performed a diachronic analysis of sound and language in toy commercials, leveraging retrieval-augmented generation (RAG) and open-weight language models in low-resource settings. A pool of 2508 UK toy advertisements spanning 14 years was semi-automatically annotated, integrating thematic coding of transcripts with audio annotation. With our RAG pipeline, we thematically coded and classified commercials by gender-target audience (feminine, masculine, or mixed) achieving *substantial* inter-coder reliability. In parallel, a music-focused multitask model was applied to annotate affective and mid-level musical perceptual attributes, enabling multimodal discourse analysis. Our findings reveal significant diachronic shifts and enduring patterns. Soundtracks classified as energizing registered an overall increase across distinct themes and audiences, but such increase was steeper for masculine-adjacent commercials. Moreover, themes stereotypically associated with masculinity paired more frequently with louder, distorted, and aggressive music, while stereotypically feminine themes with softer, calmer, and more harmonious soundtracks.

## Structure of ground-truth data
The ground-truth has the following relevant columns:

    - 'target_of_toy_ad',
    - 'voice_type', 'voice_age', 'voice_gender', 'voice_exagg', (only audio-related tasks)
    - 'transcript', 
    - CUES COLUMNS:
        - 'domesticity_nurturing_cues','consumerism_cues' (noisy), 'apparence_beauty_cues',
          'nature_animals_cues', 'love_tenderness_cues', 'magic_fantasy_cues',
          'action_adventure_cues','fight_combat_cues', 'speed_racing_cues', 
          'horror_monsters_cues','science_technology_cues (noisy)', 'sports_cues (noisy)', '
          'gross-out_cues' (noisy),'fun_play_cues (noisy)', 'arts_crafts_cues'

Most data is self explanatory (if not explained in the paper), whilst the CUED COLUMNS need some explanation. Each column contains a list of cues to the related theme that were manually annotated in the dataset. When a cue consists of more than a single word, it is separated by an underscore. The cues are not mutually exclusive, meaning that a single ad can contain cues from multiple themes. The cues are not exhaustive, meaning that an ad can contain cues that are not present in the dataset. The cues are not necessarily exclusive to the theme they are assigned to, meaning that the same cue can be present in multiple themes.

The themes are instead defined under `definitions/themes.json`.


## Prompting structure
Each of the tasks ("cues_detection", "cues_correction", "target_classification") refers to the prompts and has the following structure:

NB: "cues_correction" was not used in the final version of the experimental setup, as it degraded the quality of the output.

```
prompts
├── task_name
│   ├── prompt.txt
│   ├── inputs.json
|   ├── placeholder.json
|   └── reasoning_structure.json (optional, for now only in "target_classification")
```

- `prompt.txt` contains the prompt for the task.
- `inputs.json` contains the list of inputs for prompt completion.
- `reasoning_structure.json` contains the reasoning structure for the task.
- `placeholder.json` contains the placeholder for the task (for debugging and formatting purposes).
