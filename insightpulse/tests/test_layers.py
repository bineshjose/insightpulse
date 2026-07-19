"""Unit tests for the five layer implementations (demo AND production).

External dependencies are always faked: SQL runs against a temporary
SQLite database, the LLM engine gets a scripted router, and clustering
runs on synthetic blobs — no test touches a network or a provider key.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from insightpulse.analytics.insights import (
    BasicInsightEngine,
    DemographicBreakdownEngine,
    DriftDetector,
    FullInsightEngine,
    ResultAggregator,
)
from insightpulse.config.settings import (
    DataLayerConfig,
    EmbeddingConfig,
    GenerationConfig,
    InsightConfig,
)
from insightpulse.core.exceptions import (
    CalibrationError,
    CircuitBreakerOpenError,
    DataLayerError,
    EmbeddingError,
    GenerationError,
    InsightError,
)
from insightpulse.data.repositories import CSVRepository, SQLRepository
from insightpulse.ml.calibration import (
    EmpiricalDistributionLoader,
    FairnessConstraintManager,
    SimpleCalibrationEngine,
)
from insightpulse.ml.embeddings import (
    ClusteringEngine,
    DemographicEncoder,
    FAISSIndexManager,
    PrecomputedEmbeddingEngine,
    PurchaseTokenizer,
    TransformerEmbeddingEngine,
)
from insightpulse.ml.generation import (
    CircuitBreaker,
    DemoGenerationEngine,
    LLMGenerationEngine,
    PersonaPromptBuilder,
    ResponseParser,
)

LIKERT = ["Not at all important", "Slightly important", "Moderately important",
          "Very important", "Extremely important"]
QUESTION = {
    "question_id": "q_organic",
    "text": "How important is organic labeling when purchasing snacks?",
    "question_type": "likert_5",
    "options": LIKERT,
}


@pytest.fixture
def cohort() -> pd.DataFrame:
    """Small deterministic cohort with the attributes every layer reads."""
    rows = []
    archetypes = ["value_seeker", "premium_loyalist", "health_conscious"]
    for i in range(9):
        rows.append({
            "panelist_id": f"HH{i:05d}",
            "age_group": ["25-34", "45-54", "65+"][i % 3],
            "income_group": ["low", "middle", "high"][i % 3],
            "region": ["south", "west", "midwest"][i % 3],
            "household_size": "2",
            "education_level": "bachelors",
            "employment_status": "employed",
            "has_children": i % 2 == 0,
            "behavioral_archetype": archetypes[i % 3],
            "cluster_id": i % 3,
        })
    return pd.DataFrame(rows)


# ===========================================================================
# L1 — Data layer
# ===========================================================================

class TestCSVRepository:
    async def test_loads_and_validates_all_datasets(self):
        repo = CSVRepository()
        assert len(await repo.get_panelists()) == 2_560
        assert len(await repo.get_purchases()) == 27_520
        assert len(await repo.get_survey_responses()) == 12_800

    async def test_missing_file_raises(self, tmp_path):
        repo = CSVRepository(data_dir=tmp_path)
        with pytest.raises(DataLayerError, match="not found"):
            await repo.get_panelists()

    async def test_invalid_rows_dropped_within_tolerance(self, tmp_path):
        frame = pd.DataFrame([
            {"panelist_id": "A", "age_group": "25-34", "income_group": "middle",
             "region": "south", "household_size": "2", "education_level": "bachelors",
             "employment_status": "employed", "has_children": True},
        ] * 19 + [
            {"panelist_id": "BAD", "age_group": "not-an-age", "income_group": "middle",
             "region": "south", "household_size": "2", "education_level": "bachelors",
             "employment_status": "employed", "has_children": True},
        ])
        frame.to_csv(tmp_path / "panelists.csv", index=False)
        repo = CSVRepository(
            data_dir=tmp_path,
            config=DataLayerConfig(max_invalid_row_fraction=0.10),
        )
        loaded = await repo.get_panelists()
        assert len(loaded) == 19  # bad row dropped, load succeeded

    async def test_too_many_invalid_rows_fail_the_load(self, tmp_path):
        frame = pd.DataFrame([
            {"panelist_id": "BAD", "age_group": "nope", "income_group": "x",
             "region": "y", "household_size": "2", "education_level": "b",
             "employment_status": "employed", "has_children": True},
        ] * 10)
        frame.to_csv(tmp_path / "panelists.csv", index=False)
        repo = CSVRepository(data_dir=tmp_path, config=DataLayerConfig())
        with pytest.raises(DataLayerError, match="failed schema validation"):
            await repo.get_panelists()

    async def test_data_version_changes_when_file_changes(self, tmp_path):
        source = CSVRepository()
        panelists = await source.get_panelists()
        panelists.to_csv(tmp_path / "panelists.csv", index=False)
        repo = CSVRepository(data_dir=tmp_path)
        version_before = await repo.get_data_version()
        panelists.head(100).to_csv(tmp_path / "panelists.csv", index=False)
        assert await repo.get_data_version() != version_before

    async def test_cache_serves_repeat_reads(self):
        repo = CSVRepository()
        first = await repo.get_panelists()
        second = await repo.get_panelists()
        assert first is second  # same cached object, no re-read


class TestSQLRepository:
    @pytest.fixture
    async def sqlite_url(self, tmp_path) -> str:
        """Provision a real (temporary) SQLite database with panel tables."""
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        url = f"sqlite+aiosqlite:///{tmp_path}/panel.db"
        engine = create_async_engine(url)
        async with engine.begin() as conn:
            await conn.execute(text(
                "CREATE TABLE panelists (panelist_id TEXT, age_group TEXT, "
                "income_group TEXT, region TEXT, household_size TEXT, "
                "education_level TEXT, employment_status TEXT, has_children BOOLEAN)"
            ))
            await conn.execute(text(
                "INSERT INTO panelists VALUES "
                "('HH1','25-34','middle','south','2','bachelors','employed',1), "
                "('HH2','45-54','high','west','3-4','masters','employed',0)"
            ))
            await conn.execute(text(
                "CREATE TABLE purchases (panelist_id TEXT, transaction_date TEXT, "
                "product_category TEXT, brand TEXT, quantity INTEGER, "
                "unit_price REAL, total_value REAL, store_type TEXT, "
                "is_promotion BOOLEAN)"
            ))
            await conn.execute(text(
                "INSERT INTO purchases VALUES "
                "('HH1','2026-01-15','snacks','CrispWave',2,3.5,7.0,'supermarket',0)"
            ))
            await conn.execute(text(
                "CREATE TABLE survey_responses (response_id TEXT, question_id TEXT, "
                "panelist_id TEXT, answer TEXT, answer_index INTEGER)"
            ))
            await conn.execute(text(
                "INSERT INTO survey_responses VALUES "
                "('R1','q_organic','HH1','Very important',3)"
            ))
        await engine.dispose()
        return url

    async def test_loads_all_tables(self, sqlite_url):
        repo = SQLRepository(database_url=sqlite_url)
        assert len(await repo.get_panelists()) == 2
        assert len(await repo.get_purchases()) == 1
        assert len(await repo.get_survey_responses()) == 1

    async def test_data_version_from_row_counts(self, sqlite_url):
        repo = SQLRepository(database_url=sqlite_url)
        version = await repo.get_data_version()
        assert isinstance(version, str) and len(version) == 16

    async def test_unreachable_db_degrades_to_fallback(self):
        repo = SQLRepository(
            database_url="postgresql+asyncpg://nobody@127.0.0.1:1/nothing",
            config=DataLayerConfig(retry_attempts=1, retry_wait_seconds=0.0),
            fallback=CSVRepository(),
        )
        assert len(await repo.get_panelists()) == 2_560  # served by fallback

    async def test_unreachable_db_without_fallback_raises(self):
        repo = SQLRepository(
            database_url="postgresql+asyncpg://nobody@127.0.0.1:1/nothing",
            config=DataLayerConfig(retry_attempts=1, retry_wait_seconds=0.0),
        )
        with pytest.raises(DataLayerError):
            await repo.get_panelists()


# ===========================================================================
# L2 — Embedding layer
# ===========================================================================

class TestDemographicEncoder:
    def test_deterministic_and_correct_dimension(self, cohort):
        encoder = DemographicEncoder()
        row = cohort.iloc[0].to_dict()
        first, second = encoder.encode_row(row), encoder.encode_row(row)
        assert first.shape == (encoder.dim,)
        assert np.array_equal(first, second)

    def test_ordinal_attributes_preserve_order(self):
        encoder = DemographicEncoder()
        young = encoder.encode_row({"age_group": "18-24"})
        old = encoder.encode_row({"age_group": "65+"})
        assert young[0] < old[0]  # age is ordinal, order must survive


class TestPurchaseTokenizer:
    @pytest.fixture
    def purchases(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"panelist_id": "HH1", "transaction_date": "2026-01-01",
             "product_category": "snacks", "unit_price": 3.0, "is_promotion": False},
            {"panelist_id": "HH1", "transaction_date": "2026-01-02",
             "product_category": "beverages", "unit_price": 12.0, "is_promotion": True},
        ])

    def test_fit_builds_vocabulary(self, purchases):
        tokenizer = PurchaseTokenizer(max_length=8).fit(purchases)
        assert tokenizer.vocab_size == 4  # 2 tokens + PAD + OOV

    def test_encode_pads_and_orders(self, purchases):
        tokenizer = PurchaseTokenizer(max_length=8).fit(purchases)
        ids = tokenizer.encode_sequence(purchases)
        assert ids.shape == (8,)
        assert (ids[2:] == 0).all()  # PAD after the two events

    def test_unseen_token_maps_to_oov(self, purchases):
        tokenizer = PurchaseTokenizer(max_length=4).fit(purchases)
        unseen = pd.DataFrame([{
            "panelist_id": "HH9", "transaction_date": "2026-02-01",
            "product_category": "never_seen", "unit_price": 1.0,
            "is_promotion": False,
        }])
        assert tokenizer.encode_sequence(unseen)[0] == 1  # OOV id

    def test_truncation_keeps_most_recent(self, purchases):
        tokenizer = PurchaseTokenizer(max_length=1).fit(purchases)
        ids = tokenizer.encode_sequence(purchases)
        # The later (beverages) event survives truncation.
        beverages_token = tokenizer.encode_sequence(purchases.iloc[[1]])[0]
        assert ids[0] == beverages_token

    def test_encode_before_fit_raises(self, purchases):
        with pytest.raises(EmbeddingError, match="fitted"):
            PurchaseTokenizer(max_length=4).encode_sequence(purchases)


class TestClusteringEngine:
    def test_scan_recovers_true_cluster_count(self):
        rng = np.random.default_rng(0)
        blobs = {}
        for c in range(3):  # 3 well-separated blobs
            center = np.zeros(16)
            center[c] = 10.0
            for i in range(30):
                blobs[f"p{c}_{i}"] = center + rng.normal(0, 0.1, 16)
        result = ClusteringEngine(
            EmbeddingConfig(kmeans_k_min=2, kmeans_k_max=5)
        ).run(blobs, scan_k=True)
        assert result.chosen_k == 3
        assert result.silhouette > 0.9
        assert set(result.silhouette_by_k) == {2, 3, 4, 5}

    def test_too_few_samples_raises(self):
        embeddings = {"a": np.zeros(4), "b": np.ones(4)}
        with pytest.raises(EmbeddingError, match="clusters"):
            ClusteringEngine(EmbeddingConfig(num_clusters=5)).run(
                embeddings, scan_k=False
            )


class TestPrecomputedEmbeddingEngine:
    @pytest.fixture
    def purchases(self, cohort) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        rows = []
        for pid in cohort["panelist_id"]:
            for _ in range(10):
                rows.append({
                    "panelist_id": pid, "transaction_date": "2026-01-01",
                    "product_category": rng.choice(["snacks", "produce"]),
                    "unit_price": float(rng.uniform(1, 20)),
                    "total_value": 5.0, "is_promotion": bool(rng.random() < 0.3),
                })
        return pd.DataFrame(rows)

    def test_encode_shapes_and_normalization(self, cohort, purchases):
        engine = PrecomputedEmbeddingEngine(EmbeddingConfig(embedding_dim=32))
        embeddings = engine.encode(purchases, cohort)
        assert len(embeddings) == len(cohort)
        vector = next(iter(embeddings.values()))
        assert vector.shape == (32,)
        assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)

    def test_disk_cache_roundtrip(self, cohort, purchases, tmp_path):
        config = EmbeddingConfig(embedding_dim=32)
        first = PrecomputedEmbeddingEngine(
            config, cache_dir=tmp_path, data_version="v1"
        ).encode(purchases, cohort)
        # Second engine must serve identical vectors from the cache file.
        second = PrecomputedEmbeddingEngine(
            config, cache_dir=tmp_path, data_version="v1"
        ).encode(purchases.iloc[0:0], cohort)  # no purchases needed on hit
        assert all(np.array_equal(first[k], second[k]) for k in first)

    def test_stale_cache_invalidated_by_version(self, cohort, purchases, tmp_path):
        config = EmbeddingConfig(embedding_dim=32)
        PrecomputedEmbeddingEngine(
            config, cache_dir=tmp_path, data_version="v1"
        ).encode(purchases, cohort)
        engine = PrecomputedEmbeddingEngine(
            config, cache_dir=tmp_path, data_version="v2"
        )
        assert engine._load_cache() is None

    def test_find_similar_excludes_query(self, cohort, purchases):
        engine = PrecomputedEmbeddingEngine(EmbeddingConfig(embedding_dim=32))
        embeddings = engine.encode(purchases, cohort)
        query = str(cohort["panelist_id"].iloc[0])
        neighbors = engine.find_similar(query, 3, embeddings)
        assert len(neighbors) == 3
        assert all(pid != query for pid, _ in neighbors)

    def test_conditioning_vector_dimension(self, cohort, purchases):
        engine = PrecomputedEmbeddingEngine(EmbeddingConfig(embedding_dim=32))
        embeddings = engine.encode(purchases, cohort)
        vectors = engine.build_conditioning_vectors(cohort, embeddings)
        assert next(iter(vectors.values())).shape == (32 + DemographicEncoder().dim,)

    def test_missing_embedding_raises(self, cohort):
        engine = PrecomputedEmbeddingEngine(EmbeddingConfig(embedding_dim=32))
        with pytest.raises(EmbeddingError, match="No behavioral embedding"):
            engine.build_conditioning_vectors(cohort, {})


class TestTransformerEmbeddingEngine:
    @pytest.fixture
    def purchases(self, cohort) -> pd.DataFrame:
        rng = np.random.default_rng(2)
        rows = []
        for pid in cohort["panelist_id"]:
            for day in range(5):
                rows.append({
                    "panelist_id": pid, "transaction_date": f"2026-01-{day+1:02d}",
                    "product_category": rng.choice(["snacks", "dairy"]),
                    "unit_price": float(rng.uniform(1, 25)),
                    "total_value": 4.0, "is_promotion": False,
                })
        return pd.DataFrame(rows)

    def _config(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            embedding_dim=16, encoder_hidden_dim=32, encoder_num_heads=2,
            encoder_num_layers=1, max_sequence_length=8,
        )

    def test_encode_produces_unit_norm_embeddings(self, cohort, purchases):
        engine = TransformerEmbeddingEngine(self._config())
        embeddings = engine.encode(purchases, cohort)
        assert len(embeddings) == len(cohort)
        for vector in embeddings.values():
            assert vector.shape == (16,)
            assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-4)

    def test_seeded_init_is_reproducible(self, cohort, purchases):
        first = TransformerEmbeddingEngine(self._config()).encode(purchases, cohort)
        second = TransformerEmbeddingEngine(self._config()).encode(purchases, cohort)
        for key in first:
            assert np.allclose(first[key], second[key], atol=1e-6)

    def test_faiss_index_roundtrip(self, tmp_path):
        config = EmbeddingConfig(embedding_dim=16, faiss_distance_threshold=100.0)
        rng = np.random.default_rng(3)
        embeddings = {f"p{i}": rng.standard_normal(16).astype(np.float32)
                      for i in range(20)}
        manager = FAISSIndexManager(config)
        manager.build(embeddings)  # 20 points -> flat fallback, logged
        hits = manager.query(embeddings["p0"], 3)
        assert hits[0][0] == "p0" and hits[0][1] < 1e-5  # self is nearest

        path = tmp_path / "test.index"
        manager.save(path)
        restored = FAISSIndexManager(config)
        restored.load(path)
        assert restored.query(embeddings["p0"], 1)[0][0] == "p0"

    def test_query_before_build_raises(self):
        manager = FAISSIndexManager(EmbeddingConfig())
        with pytest.raises(EmbeddingError, match="not built"):
            manager.query(np.zeros(8), 1)


# ===========================================================================
# L3 — Generative layer
# ===========================================================================

class TestResponseParser:
    def test_fallback_chain_order(self):
        parser = ResponseParser()
        cases = {
            "json": json.dumps(
                {"answer": "Very important", "reasoning": "r", "confidence": 0.9}
            ),
            "regex": 'Sure! {"answer": "Very important", "confidence": 0.7',
            "option_scan": "I'd say moderately important overall.",
            "raw": "no idea what to pick here",
        }
        for expected, content in cases.items():
            assert parser.parse(content, QUESTION)["parse_method"] == expected

    def test_confidence_clamped(self):
        parsed = ResponseParser().parse(
            json.dumps({"answer": "Very important", "confidence": 4.2}), QUESTION
        )
        assert parsed["confidence"] == 1.0

    def test_ambiguous_option_scan_falls_through_to_raw(self):
        text = "Either slightly important or very important."
        assert ResponseParser().parse(text, QUESTION)["parse_method"] == "raw"

    def test_markdown_fences_stripped(self):
        fenced = '```json\n{"answer": "Very important", "confidence": 0.8}\n```'
        assert ResponseParser().parse(fenced, QUESTION)["parse_method"] == "json"


class TestCircuitBreaker:
    def test_full_state_machine(self):
        now = [0.0]
        breaker = CircuitBreaker(threshold=2, recovery_seconds=10, clock=lambda: now[0])
        assert breaker.state == "closed"

        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "open"
        with pytest.raises(CircuitBreakerOpenError):
            breaker.before_call()

        now[0] = 11.0
        assert breaker.state == "half_open"
        breaker.before_call()  # single probe admitted
        with pytest.raises(CircuitBreakerOpenError):
            breaker.before_call()  # second concurrent probe shed

        breaker.record_success()
        assert breaker.state == "closed"

    def test_half_open_failure_reopens(self):
        now = [0.0]
        breaker = CircuitBreaker(threshold=1, recovery_seconds=5, clock=lambda: now[0])
        breaker.record_failure()
        now[0] = 6.0
        breaker.before_call()
        breaker.record_failure()  # probe failed
        assert breaker.state == "open"


class TestPersonaPromptBuilder:
    def test_prompt_carries_demographics_and_version(self, cohort):
        builder = PersonaPromptBuilder(GenerationConfig())
        system, user = builder.build(cohort.iloc[0].to_dict(), QUESTION)
        assert "25-34" in system
        assert builder.template_version in system
        assert QUESTION["text"] in user

    def test_prior_answers_injected(self, cohort):
        builder = PersonaPromptBuilder(GenerationConfig())
        _, user = builder.build(
            cohort.iloc[0].to_dict(), QUESTION, prior_responses=["Very important"]
        )
        assert "previous answers" in user
        assert "Very important" in user


class TestDemoGenerationEngine:
    async def test_deterministic_for_same_seed(self, cohort):
        engine = DemoGenerationEngine()
        first = await engine.generate_responses([QUESTION], cohort, "gpt-4o", seed=7)
        second = await engine.generate_responses([QUESTION], cohort, "gpt-4o", seed=7)
        assert [r["answer"] for r in first] == [r["answer"] for r in second]

    async def test_unknown_model_raises(self, cohort):
        with pytest.raises(GenerationError, match="No demo profile"):
            await DemoGenerationEngine().generate_responses(
                [QUESTION], cohort, "made-up-model"
            )

    async def test_answers_come_from_options(self, cohort):
        responses = await DemoGenerationEngine().generate_responses(
            [QUESTION], cohort, "claude-sonnet-4-6"
        )
        assert len(responses) == len(cohort)
        assert all(r["answer"] in LIKERT for r in responses)


class _ScriptedRouter:
    """Router double: yields scripted results/exceptions in order."""

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.prompts: list[str] = []

    async def generate(self, prompt: str = "", **_: Any) -> Any:
        self.prompts.append(prompt)
        item = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(item, Exception):
            raise item
        return item


class _RouterResult:
    def __init__(self, content: str, tokens: int = 100) -> None:
        self.content = content
        self.total_tokens = tokens
        self.cost_usd = 0.001
        self.latency_ms = 3.0


class TestLLMGenerationEngine:
    def _engine(self, router: Any, **config_overrides: Any) -> LLMGenerationEngine:
        config = GenerationConfig(
            retry_attempts=2, retry_wait_seconds=0.0, **config_overrides
        )
        return LLMGenerationEngine(config=config, router=router)

    async def test_happy_path_with_provenance(self, cohort):
        router = _ScriptedRouter([_RouterResult(
            json.dumps({"answer": "Very important", "confidence": 0.9})
        )])
        responses = await self._engine(router).generate_responses(
            [QUESTION], cohort.head(3), "claude-sonnet-4-6"
        )
        assert len(responses) == 3
        assert all(r["parse_method"] == "json" for r in responses)
        assert all(r["model_used"] == "claude-sonnet-4-6" for r in responses)

    async def test_retry_recovers_from_transient_failure(self, cohort):
        router = _ScriptedRouter([
            RuntimeError("transient"),
            _RouterResult(json.dumps({"answer": "Very important"})),
        ])
        responses = await self._engine(router).generate_responses(
            [QUESTION], cohort.head(1), "gpt-4o"
        )
        assert len(responses) == 1  # second attempt succeeded

    async def test_all_failures_raise_generation_error(self, cohort):
        router = _ScriptedRouter([RuntimeError("down")])
        with pytest.raises(GenerationError):
            await self._engine(router).generate_responses(
                [QUESTION], cohort.head(2), "gpt-4o"
            )

    async def test_token_budget_flagged(self, cohort):
        router = _ScriptedRouter([_RouterResult(
            json.dumps({"answer": "Very important"}), tokens=9000
        )])
        responses = await self._engine(router).generate_responses(
            [QUESTION], cohort.head(1), "claude-sonnet-4-6"
        )
        assert responses[0]["over_token_budget"] is True

    async def test_sequential_prior_answers_reach_prompts(self, cohort):
        router = _ScriptedRouter([_RouterResult(
            json.dumps({"answer": "Very important"})
        )])
        follow_up = {**QUESTION, "question_id": "q2", "text": "Pay more for it?"}
        await self._engine(router).generate_responses(
            [QUESTION, follow_up], cohort.head(1), "claude-sonnet-4-6"
        )
        assert "previous answers" not in router.prompts[0]
        assert "previous answers" in router.prompts[1]
        assert "Very important" in router.prompts[1]


# ===========================================================================
# L4 — Calibration layer (solver details in test_calibration.py)
# ===========================================================================

class TestCalibrationLayer:
    async def test_loader_uses_empirical_bank(self):
        loader = EmpiricalDistributionLoader(CSVRepository())
        target = await loader.load_target("q_organic", LIKERT)
        assert np.isclose(target.sum(), 1.0)
        assert not np.allclose(target, 0.2)  # real data is not uniform

    async def test_loader_uniform_fallback_for_unknown_question(self):
        loader = EmpiricalDistributionLoader(CSVRepository())
        target = await loader.load_target("q_never_asked", LIKERT)
        assert np.allclose(target, 0.2)

    async def test_calibrate_question_without_loader_raises(self):
        engine = SimpleCalibrationEngine(target_loader=None)
        with pytest.raises(CalibrationError, match="loader"):
            await engine.calibrate_question("q", LIKERT, np.full(5, 0.2))

    async def test_demo_engine_improves_and_checks_fairness(self):
        engine = SimpleCalibrationEngine(
            target_loader=EmpiricalDistributionLoader(CSVRepository())
        )
        skewed_group = np.array([0.9, 0.04, 0.03, 0.02, 0.01])
        output, metrics = await engine.calibrate_question(
            "q_organic", LIKERT, np.array([0.6, 0.2, 0.1, 0.06, 0.04]),
            group_distributions={"65+": skewed_group},
        )
        assert output.converged is True
        assert metrics.js_divergence_after < metrics.js_divergence_before
        assert "65+" in metrics.demographic_parity_before

    def test_fairness_manager_only_repairs_violations(self):
        manager = FairnessConstraintManager(lambda_fairness=0.5, tolerance=0.3)
        overall = np.full(5, 0.2)
        groups = {
            "ok": np.array([0.25, 0.2, 0.2, 0.2, 0.15]),
            "violating": np.array([0.9, 0.04, 0.03, 0.02, 0.01]),
        }
        adjusted, before, after = manager.enforce(groups, overall)
        assert np.array_equal(adjusted["ok"], groups["ok"])
        assert after["violating"] < before["violating"]


# ===========================================================================
# L5 — Insight layer
# ===========================================================================

def _make_responses(cohort: pd.DataFrame, answers: list[str]) -> list[dict[str, Any]]:
    rows = cohort.to_dict(orient="records")
    return [
        {
            "question_id": QUESTION["question_id"],
            "panelist_id": row["panelist_id"],
            "answer": answers[i % len(answers)],
            "confidence": 0.8,
            "is_valid": True,
            "validation_flags": [],
            "age_group": row["age_group"],
            "income_group": row["income_group"],
            "region": row["region"],
        }
        for i, row in enumerate(rows)
    ]


class TestResultAggregator:
    def test_distribution_and_entropy(self, cohort):
        responses = _make_responses(cohort, ["Very important", "Slightly important"])
        results = ResultAggregator().aggregate([QUESTION], responses)
        (result,) = results
        counted = {d["option"]: d["count"] for d in result["distribution"]}
        assert counted["Very important"] == 5
        assert counted["Slightly important"] == 4
        assert 0.9 < result["entropy"] <= 1.0  # near-even two-way split

    def test_summary_rates(self, cohort):
        responses = _make_responses(cohort, ["Very important"])
        responses[0]["validation_flags"] = ["hallucination_detected"]
        summary = ResultAggregator().summarize(responses)
        assert summary["total_responses"] == len(cohort)
        assert summary["hallucination_rate"] == pytest.approx(1 / len(cohort), abs=1e-4)


class TestDemographicBreakdownEngine:
    def test_detects_engineered_dependence(self):
        # 300 responses where the answer is fully determined by age group —
        # chi-square must flag dependence as significant.
        rows = []
        for i in range(300):
            age = "25-34" if i % 2 == 0 else "65+"
            answer = "Very important" if age == "25-34" else "Not at all important"
            rows.append({
                "question_id": QUESTION["question_id"], "answer": answer,
                "age_group": age, "is_valid": True, "validation_flags": [],
            })
        breakdown = DemographicBreakdownEngine(InsightConfig()).breakdown(
            QUESTION, rows
        )
        entry = breakdown["age_group"]
        assert entry["testable"] is True
        assert entry["significant"] is True

    def test_sparse_table_flagged_not_testable(self, cohort):
        responses = _make_responses(cohort, ["Very important", "Slightly important"])
        breakdown = DemographicBreakdownEngine(InsightConfig()).breakdown(
            QUESTION, responses
        )
        entry = breakdown["age_group"]  # 9 responses over 3 groups: sparse
        assert entry["testable"] is False
        assert entry["significant"] is None


class TestDriftDetector:
    @staticmethod
    def _purchases(shift_last: int = 0) -> pd.DataFrame:
        """10 months of two-category purchases; optionally shift the tail."""
        rows = []
        for month in range(1, 11):
            shifted = month > 10 - shift_last
            for i in range(200):
                category = (
                    "produce" if (i % 10 < (8 if shifted else 5)) else "snacks"
                )
                rows.append({
                    "transaction_date": f"2026-{month:02d}-15",
                    "product_category": category,
                })
        return pd.DataFrame(rows)

    def test_stationary_data_does_not_fire(self):
        report = DriftDetector(InsightConfig()).detect(self._purchases())
        assert report.fired is False
        assert len(report.series) == 7  # 10 months minus 3 baseline

    def test_injected_shift_fires_trigger(self):
        config = InsightConfig(drift_sigma_multiplier=1.0)
        report = DriftDetector(config).detect(self._purchases(shift_last=3))
        assert report.fired is True
        assert report.rule in {"hard", "consecutive"}
        assert "Retrain" in report.recommendation

    def test_too_few_periods_raises(self):
        frame = pd.DataFrame([
            {"transaction_date": "2026-01-01", "product_category": "snacks"},
            {"transaction_date": "2026-02-01", "product_category": "snacks"},
        ])
        with pytest.raises(InsightError, match="periods"):
            DriftDetector(InsightConfig()).detect(frame)


class TestInsightEngines:
    def test_basic_engine_reports_essentials(self, cohort):
        responses = _make_responses(cohort, ["Very important"])
        report = BasicInsightEngine().analyze([QUESTION], responses)
        assert report["summary"]["total_responses"] == len(cohort)
        assert "demographic_breakdowns" not in report

    def test_full_engine_adds_breakdowns(self, cohort):
        responses = _make_responses(cohort, ["Very important", "Slightly important"])
        report = FullInsightEngine().analyze([QUESTION], responses)
        assert "demographic_breakdowns" in report
        assert QUESTION["question_id"] in report["demographic_breakdowns"]
