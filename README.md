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

## **CORE COMPONENTS**

### **A. Configuration & Argument Parsing** (dynamic_parser.py)
- **Purpose**: Load and merge configuration from multiple sources
- **Inputs**: 
  - Command-line arguments
  - parser.json (argument definitions)
  - base_parameters.json (default parameters)
  - themes.json (10 thematic categories)
  - main_task.json (target classification: Girls/women, Mixed, Boys/men)
- **Output**: `argparse.Namespace` with all configuration merged

### **B. State Management** (states.py)
- **GraphState**: The flow state carrying:
  - `current_id`: datapoint identifier
  - `current_transcript`: ad text to classify
  - `collected_cues`: theme detections (dict of ThemeState → List[str])
  - `target_class`: final gender prediction
  - `exclude_from_vector_store`: IDs to skip (for cross-validation)

### **C. Data Layer** (tools.py)
- **`load_data()`**: Loads CSV groundtruth with theme cues and target labels
- **`convert_df_to_documents()`**: Converts dataframe rows to LangChain `Document` objects with metadata
  - Metadata includes: themes, cue words, target classification
- **`get_prompt()`**: Loads prompt templates, inputs, and reasoning structures from prompts subfolder
- **`get_top_classes_per_theme()`**: Identifies dominant classes per theme using frequency balancing

### **D. Vector Database** (vector_db.py)
- **VectorDB class**: Wrapper around Chroma with HuggingFace embeddings
- **Key Methods**:
  - `create_vector_database()`: Embeds and stores documents
  - `query_vector_database()`: Similarity search with metadata filtering
  - `get_positive_negative_examples()`: Retrieves examples by theme presence/absence
  - `get_targeted_examples()`: Retrieves class-specific examples for few-shot learning

### **E. LLM Interface** (llm.py)
- **Chain class**: Wraps HuggingFace models with LangChain
- **Key Methods**:
  - `run()`: Executes prompt + model with error recovery
  - Handles JSON parsing with fallback corrections
  - Uses task-specific placeholders on parsing failure
- **Error Handling**: 3-tier retry mechanism
  1. Parse LLM output as JSON
  2. Ask model to reformat if invalid
  3. Return placeholder if both fail

### **F. Workflow Orchestration** (workflow.py)
- **WorkflowHandler class**: Main orchestrator using LangGraph StateGraph
- **Workflow Graph**:
  ```
  START → _initialize_datapoint → _detect_cues → _classify_target → END
  ```

---

## **WORKFLOW NODES (Processing Pipeline)**

### **Node 1: _initialize_datapoint()**
- Initializes empty `collected_cues` dict for all themes
- Creates initial GraphState with transcript and ID

### **Node 2: _detect_cues()**
- **Loop**: For each theme in themes.json
- **RAG**: Retrieves positive/negative examples from vector store
- **LLM Call**: Uses prompt from cues_detection + Chain.run()
- **Output**: Populates `collected_cues[theme]` with detected cue words
- **Optional**: Checks precomputed results from `old_run_folder`

### **Node 3: _classify_target()**
- **Theme Filter**: Only uses themes where cues were detected
- **Parallel Subtasks** (via `__subtask_wrapper`):
  - For subsets of target classes, retrieves targeted examples
  - Calls LLM with reasoning structure from reasoning_structure.json
- **Decision Logic**: `decision_logic()` combines multiple task results via voting
- **Output**: `target_class` (Girls/women, Mixed, or Boys/men)

### **Helper: decision_logic()**
- Counts predictions from parallel subtasks
- Returns majority vote
- On tie: uses task-specific class sets to break ties
- Normalizes output (e.g., "boys" → "Boys/men")

---

## **DATA RELATIONSHIPS**

```
Configuration (parser.json + base_parameters.json)
    ↓
Themes Definition (themes.json) + Main Task (main_task.json)
    ↓
Groundtruth Data (clean_groundtruth.csv)
    ↓
Documents + Metadata (via convert_df_to_documents)
    ↓
Vector Store (Chroma + HF Embeddings)
    ↓
RAG Examples (Retrieved for each task)
    ↓
Prompt Templates (from prompts/ folder)
    ↓
LLM Chains (with error recovery)
    ↓
GraphState (Accumulated through workflow)
    ↓
Output (Results saved to output_folder)
```

---

### **EXECUTION FLOW (app.py)**

```
Parse Arguments
    ↓
Load/Create Vector Store (from groundtruth data)
    ↓
Load Models:
    - Cues Detection (model_name_cd)
    - Target Classification (model_name_tc, may reuse)
    ↓
Create Chains (LLM wrappers)
    ↓
Build WorkflowHandler (compile StateGraph)
    ↓
For Each Unseen Datapoint:
    - Prepare music predictions (optional)
    - Stream initial state through workflow
    - Save results to output_folder
```

