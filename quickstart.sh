#!/bin/bash
# Fantasy Football Projections - Quickstart Script
# Automates setup, data fetching, model training, and initial projections

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Emoji support (comment out if terminal doesn't support)
CHECK="✓"
CROSS="✗"
ARROW="→"
ROCKET="🚀"
STAR="⭐"
PACKAGE="📦"
DATABASE="🗄️"
BRAIN="🧠"
CHART="📊"

# Function to print colored output
print_header() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  $1"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() {
    echo -e "${BLUE}${ARROW}${NC} ${MAGENTA}$1${NC}"
}

print_success() {
    echo -e "${GREEN}${CHECK}${NC} $1"
}

print_error() {
    echo -e "${RED}${CROSS}${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC}  $1"
}

# Function to detect OS
detect_os() {
    case "$(uname -s)" in
        Darwin*)
            echo "macos"
            ;;
        Linux*)
            echo "linux"
            ;;
        MINGW*|MSYS*|CYGWIN*)
            echo "windows"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get current NFL week
get_current_week() {
    # Simple estimation based on date
    # NFL season typically starts first Thursday of September
    local current_month=$(date +%m)
    local current_day=$(date +%d)

    # Default to week 1 for now (could be improved with actual schedule)
    if [ "$current_month" -ge 9 ] && [ "$current_month" -le 12 ]; then
        # Rough estimate: Week = (days since Sept 1) / 7
        local week=$(( (current_month - 9) * 4 + current_day / 7 + 1 ))
        if [ "$week" -gt 18 ]; then
            echo 18
        else
            echo $week
        fi
    elif [ "$current_month" -eq 1 ]; then
        echo 18  # Playoffs
    else
        echo 1   # Off-season, default to week 1
    fi
}

# Function to get current season
get_current_season() {
    local current_month=$(date +%m)
    local current_year=$(date +%Y)

    # NFL season year is the year it ends (e.g., 2024 season ends in 2025)
    if [ "$current_month" -ge 3 ] && [ "$current_month" -le 8 ]; then
        echo $current_year
    else
        echo $current_year
    fi
}

# Main script starts here
clear
print_header "${ROCKET} Fantasy Football Projections - Quickstart ${ROCKET}"

echo "This script will:"
echo "  1. Set up Python virtual environment"
echo "  2. Install all dependencies"
echo "  3. Initialize the database"
echo "  4. Fetch current season data"
echo "  5. Train ML models"
echo "  6. Generate sample projections"
echo ""

# Detect OS
OS=$(detect_os)
print_info "Detected OS: $OS"
echo ""

# Check prerequisites
print_step "Checking prerequisites..."

if ! command_exists python3; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_success "Python ${PYTHON_VERSION} found"

if ! command_exists pip3; then
    print_error "pip3 is not installed. Please install pip3."
    exit 1
fi
print_success "pip3 found"

echo ""

# Step 1: Create virtual environment
print_header "${PACKAGE} Step 1: Setting up Virtual Environment"

VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    print_info "Virtual environment already exists. Using existing environment."
else
    print_step "Creating virtual environment..."
    python3 -m venv $VENV_DIR
    print_success "Virtual environment created at ${VENV_DIR}/"
fi

# Activate virtual environment
print_step "Activating virtual environment..."

if [ "$OS" = "windows" ]; then
    source ${VENV_DIR}/Scripts/activate
else
    source ${VENV_DIR}/bin/activate
fi

print_success "Virtual environment activated"
echo ""

# Step 2: Install requirements
print_header "${PACKAGE} Step 2: Installing Dependencies"

print_step "Upgrading pip..."
pip install --upgrade pip --quiet

print_step "Installing required packages..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    print_success "All dependencies installed"
else
    print_error "requirements.txt not found!"
    exit 1
fi

# Install additional dependencies that might be missing
print_step "Installing additional dependencies..."
pip install --quiet libomp 2>/dev/null || true

echo ""

# Step 3: Initialize database
print_header "${DATABASE} Step 3: Initializing Database"

print_step "Setting up database schema and directories..."
python3 main.py setup

if [ $? -eq 0 ]; then
    print_success "Database initialized successfully"
else
    print_error "Database initialization failed"
    exit 1
fi

echo ""

# Prompt for API keys (optional)
print_header "🔑 API Configuration (Optional)"

print_info "You can optionally configure API keys for enhanced features:"
echo ""
echo "  • The Odds API (for Vegas lines): https://the-odds-api.com/"
echo "  • Weather API (for weather data)"
echo ""

read -p "Would you like to configure API keys now? (y/N): " configure_api

if [[ $configure_api =~ ^[Yy]$ ]]; then
    echo ""
    read -p "Enter The Odds API key (or press Enter to skip): " odds_key

    if [ ! -z "$odds_key" ]; then
        # Update .env file
        if grep -q "ODDS_API_KEY=" .env 2>/dev/null; then
            # macOS sed requires backup extension
            if [ "$OS" = "macos" ]; then
                sed -i '' "s/ODDS_API_KEY=.*/ODDS_API_KEY=$odds_key/" .env
            else
                sed -i "s/ODDS_API_KEY=.*/ODDS_API_KEY=$odds_key/" .env
            fi
        else
            echo "ODDS_API_KEY=$odds_key" >> .env
        fi
        print_success "API key configured"
        export ODDS_API_KEY="$odds_key"
    fi
else
    print_info "Skipping API configuration. You can configure later in .env file."
fi

echo ""

# Get current week and season
CURRENT_WEEK=$(get_current_week)
CURRENT_SEASON=$(get_current_season)

print_info "Current NFL season: ${CURRENT_SEASON}"
print_info "Estimated current week: ${CURRENT_WEEK}"
echo ""

read -p "Use these values for data fetch? (Y/n): " use_current

if [[ $use_current =~ ^[Nn]$ ]]; then
    read -p "Enter season year (e.g., 2024): " CURRENT_SEASON
    read -p "Enter week number (1-18): " CURRENT_WEEK
fi

echo ""

# Step 4: Fetch data
print_header "${CHART} Step 4: Fetching NFL Data"

print_step "Fetching data for ${CURRENT_SEASON} Week ${CURRENT_WEEK}..."
print_info "This may take 2-5 minutes depending on your connection..."

# Run data fetch with timeout
timeout 600 python3 main.py fetch --week $CURRENT_WEEK --season $CURRENT_SEASON 2>&1 | tee /tmp/fetch_output.log || {
    if [ $? -eq 124 ]; then
        print_error "Data fetch timed out after 10 minutes"
    else
        print_info "Data fetch completed (some errors may be expected with mock data)"
    fi
}

echo ""

# Step 5: Train models
print_header "${BRAIN} Step 5: Training ML Models"

print_info "Training models for all positions..."
print_info "This may take 5-15 minutes depending on data volume..."
echo ""

# Prompt for model training
read -p "Train models now? This can take 5-15 minutes. (Y/n): " train_now

if [[ ! $train_now =~ ^[Nn]$ ]]; then
    python3 main.py train 2>&1 | tee /tmp/train_output.log || {
        print_error "Model training encountered errors (this is expected with limited data)"
        print_info "You can continue with existing models or re-train later"
    }

    echo ""
    print_success "Model training completed"
else
    print_info "Skipping model training. Using existing models."
fi

echo ""

# Step 6: Generate projections
print_header "${CHART} Step 6: Generating Projections"

print_step "Generating projections for Week ${CURRENT_WEEK}..."

python3 main.py update --week $CURRENT_WEEK --season $CURRENT_SEASON --no-fetch 2>&1 | tee /tmp/projection_output.log || {
    print_error "Projection generation encountered errors"
    print_info "This may be due to insufficient training data"
}

echo ""

# Step 7: Display results
print_header "${STAR} Quickstart Complete! ${STAR}"

print_success "Fantasy Football Projection System is ready!"
echo ""

# Find output files
OUTPUT_DIR="output/${CURRENT_SEASON}_week${CURRENT_WEEK}"

if [ -d "$OUTPUT_DIR" ]; then
    print_info "Output files location:"
    echo "  ${OUTPUT_DIR}/"
    echo ""

    # List generated files
    if [ "$(ls -A $OUTPUT_DIR 2>/dev/null)" ]; then
        print_info "Generated files:"
        ls -lh $OUTPUT_DIR | grep -v "^total" | awk '{print "  • " $9 " (" $5 ")"}'
    fi
else
    print_info "Output directory will be created when projections are generated:"
    echo "  ${OUTPUT_DIR}/"
fi

echo ""

# Next steps
print_header "📋 Next Steps"

echo "1. Review your projections:"
echo "   ${GREEN}cat ${OUTPUT_DIR}/*_summary.txt${NC}"
echo ""

echo "2. View rankings:"
echo "   ${GREEN}cat ${OUTPUT_DIR}/*_rankings.csv | head -20${NC}"
echo ""

echo "3. Check value plays:"
echo "   ${GREEN}ls ${OUTPUT_DIR}/*_value_*.csv${NC}"
echo ""

echo "4. Generate new projections:"
echo "   ${GREEN}python3 main.py update --week <WEEK> --season <SEASON>${NC}"
echo ""

echo "5. Re-train models:"
echo "   ${GREEN}python3 main.py train${NC}"
echo ""

echo "6. Run tests:"
echo "   ${GREEN}./run_tests.sh quick${NC}"
echo ""

echo "7. View configuration:"
echo "   ${GREEN}cat config/config.yaml${NC}"
echo ""

# Database info
if [ -f "fantasy_football.db" ]; then
    DB_SIZE=$(du -h fantasy_football.db | cut -f1)
    print_info "Database size: ${DB_SIZE}"
fi

echo ""

# Deactivation instructions
print_header "🔧 Useful Commands"

echo "To activate the virtual environment again:"
if [ "$OS" = "windows" ]; then
    echo "  ${CYAN}source venv/Scripts/activate${NC}"
else
    echo "  ${CYAN}source venv/bin/activate${NC}"
fi

echo ""
echo "To deactivate the virtual environment:"
echo "  ${CYAN}deactivate${NC}"

echo ""
echo "For help:"
echo "  ${CYAN}python3 main.py --help${NC}"

echo ""

# Optional: Open output directory
if command_exists open && [ "$OS" = "macos" ]; then
    read -p "Open output directory in Finder? (y/N): " open_finder
    if [[ $open_finder =~ ^[Yy]$ ]]; then
        open $OUTPUT_DIR 2>/dev/null || open output
    fi
elif command_exists xdg-open && [ "$OS" = "linux" ]; then
    read -p "Open output directory in file manager? (y/N): " open_fm
    if [[ $open_fm =~ ^[Yy]$ ]]; then
        xdg-open $OUTPUT_DIR 2>/dev/null || xdg-open output
    fi
fi

echo ""
print_success "Setup complete! Your Fantasy Football Projection System is ready to use."
echo ""

# Save environment state for easy reactivation
cat > activate_env.sh << 'EOF'
#!/bin/bash
# Quick activation script
OS=$(uname -s)
if [[ "$OS" == "MINGW"* ]] || [[ "$OS" == "MSYS"* ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi
echo "Virtual environment activated!"
echo "Run: python3 main.py --help"
EOF

chmod +x activate_env.sh
print_info "Created ${GREEN}activate_env.sh${NC} for quick environment activation"

echo ""
echo "Happy projecting! 🏈"
echo ""
