SOURCE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD ?= $(abspath $(SOURCE)/../../../target/paper-checkpoint)
JOB := jsr-cpu-asr
PDF := $(BUILD)/$(JOB).pdf

export SOURCE_DATE_EPOCH ?= 1795305600
export FORCE_SOURCE_DATE := 1
export LC_ALL := C
export TZ := UTC
export TEXINPUTS := $(SOURCE):$(TEXINPUTS):
export BIBINPUTS := $(SOURCE):$(BIBINPUTS):

LATEX := pdflatex -halt-on-error -interaction=nonstopmode -file-line-error -jobname=$(JOB)

.PHONY: pdf check

pdf:
	mkdir -p "$(BUILD)"
	node "$(SOURCE)/generate.mjs" "$(SOURCE)/evidence.json" "$(BUILD)"
	cd "$(BUILD)" && $(LATEX) "$(SOURCE)/paper.tex" >latex-first.log 2>&1 || { cat "$(BUILD)/latex-first.log"; exit 1; }
	cd "$(BUILD)" && bibtex "$(JOB)" >bibtex.log 2>&1 || { cat "$(BUILD)/bibtex.log"; exit 1; }
	cd "$(BUILD)" && $(LATEX) "$(SOURCE)/paper.tex" >latex-second.log 2>&1 || { cat "$(BUILD)/latex-second.log"; exit 1; }
	cd "$(BUILD)" && $(LATEX) "$(SOURCE)/paper.tex" >latex-final.log 2>&1 || { cat "$(BUILD)/latex-final.log"; exit 1; }
	@printf '%s\n' "$(PDF)"

check: pdf
	pdfinfo "$(PDF)" | awk '$$1 == "Pages:" { pages = $$2 } END { if (pages != 6) { print "Expected 6 pages, found " pages; exit 1 } }'
	pdffonts "$(PDF)" | awk 'NR > 2 && NF { fonts++; if ($$(NF-4) != "yes") { print "Unembedded font " $$1; bad = 1 } } END { exit (bad || !fonts) }'
	pdftotext -f 1 -l 1 -raw "$(PDF)" "$(BUILD)/page-one.txt"
	awk '/^Abstract—/ { abstract = 1; sub(/^Abstract—/, "") } /^Index Terms/ { abstract = 0 } abstract { words += NF } END { if (words < 150 || words > 180) { print "Expected 150–180 abstract words, found " words; exit 1 } }' "$(BUILD)/page-one.txt"
	grep -Eq '^[[:space:]]*Fig\. 1\.' "$(BUILD)/page-one.txt"
	! grep -Ei 'LaTeX Warning:.*(undefined|Rerun)|Overfull \\[hv]box' "$(BUILD)/$(JOB).log"
	! grep -E 'Warning--|error message' "$(BUILD)/$(JOB).blg"
