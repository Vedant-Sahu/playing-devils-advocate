# Playing Devil's Advocate

A multi-agent system where a teacher agent creates educational explanations that are iteratively refined through feedback from diverse student agents with different learning profiles. A game-theory reward mechanism evaluates critique quality and determines when explanations have converged to optimal clarity, ultimately validated by measuring student learning outcomes across different backgrounds.

## Team
- Teo Nocita
- Vedant Sahu

## Project Structure
```
playing-devils-advocate/
├── notebooks/
│   └── main_workflow.ipynb          # LangGraph orchestration
├── src/
│   ├── agents/
│   │   ├── coordinator_agent.py     # Orchestrates the workflow
│   │   ├── teacher_agent.py         # Generates explanations
│   │   ├── student_agent.py         # Provides critiques (multiple profiles)
│   │   ├── critique_eval_agent.py   # Validates critique quality
│   │   └── grading_agent.py         # Evaluates learning outcomes
│   └── utils/
│       ├── config.py                # Configuration and environment variables
│       └── stopping_criterion.py    # Convergence detection
├── data/
│   ├── student_profiles.json        # Student learning profiles
│   └── sample_problems.json         # Test problems
└── results/                         # Output and evaluation results
```

## Setup Instructions

1. **Clone the repository:**
   ```cmd
   git clone https://github.com/yourusername/playing-devils-advocate.git
   cd playing-devils-advocate
   ```

2. **Create and activate virtual environment:**
   ```cmd
   python -m venv venv
   venv\Scripts\activate.bat
   ```

3. **Install dependencies:**
   ```cmd
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_key_here
     ```

## How to Run

Open and run `notebooks/main_workflow.ipynb` in Jupyter:
```cmd
jupyter notebook
```

## Evaluation Metrics
- **Explanation Quality** - LLM judge evaluation
- **Learning Outcome** - Average student answer scores
- **Adaptation Effectiveness** - Performance variance across student profiles

## License
MIT License
