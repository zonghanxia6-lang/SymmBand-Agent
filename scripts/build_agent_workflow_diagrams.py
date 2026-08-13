from __future__ import annotations

from html import escape
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "topological-material-discovery-workflow-v3.svg"
INTEGRATED = ROOT / "topological-material-discovery-workflow-v3-agent-integrated.svg"
STANDALONE = ROOT / "symmband-agent-capabilities.svg"


COMMON_DEFS = """
  <defs>
    <style>
      .a-font { font-family: Arial, Helvetica, "Liberation Sans", sans-serif; fill: #263238; }
      .a-kicker { font: 700 16px Arial, Helvetica, sans-serif; letter-spacing: 2.4px; fill: #0072B2; }
      .a-title { font: 700 34px Arial, Helvetica, sans-serif; fill: #263238; }
      .a-subtitle { font: 400 18px Arial, Helvetica, sans-serif; fill: #52616B; }
      .a-card-title { font: 700 22px Arial, Helvetica, sans-serif; fill: #263238; }
      .a-body { font: 400 17px Arial, Helvetica, sans-serif; fill: #35434C; }
      .a-small { font: 400 15px Arial, Helvetica, sans-serif; fill: #52616B; }
      .a-chip { font: 700 14px Arial, Helvetica, sans-serif; letter-spacing: 0.7px; }
      .a-mono { font: 600 16px "DejaVu Sans Mono", Consolas, monospace; fill: #263238; }
      .a-on-dark { fill: #FFFFFF; }
      .a-on-dark-accent { fill: #56B4E9; }
      .a-on-dark-sub { fill: #D8E7ED; }
      .a-arrow { fill: none; stroke: #667781; stroke-width: 3.5; stroke-linecap: round; stroke-linejoin: round; marker-end: url(#agent-arrow); }
      .a-dash { fill: none; stroke: #87969F; stroke-width: 2.5; stroke-linecap: round; stroke-dasharray: 8 7; }
    </style>
    <marker id="agent-arrow" markerWidth="11" markerHeight="11" refX="9" refY="5.5" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L10,5.5 L0,11 z" fill="#667781"/>
    </marker>
    <filter id="agent-shadow" x="-10%" y="-15%" width="120%" height="135%">
      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#263238" flood-opacity="0.11"/>
    </filter>
    <linearGradient id="agent-core" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#F2FAFD"/>
      <stop offset="1" stop-color="#EFF8F4"/>
    </linearGradient>
    <pattern id="agent-dots" width="26" height="26" patternUnits="userSpaceOnUse">
      <circle cx="2" cy="2" r="1.4" fill="#B9DCEB" opacity="0.45"/>
    </pattern>
  </defs>
"""


def text(x: int, y: int, value: str, cls: str, anchor: str | None = None) -> str:
    anchor_attr = f' text-anchor="{anchor}"' if anchor else ""
    return f'<text x="{x}" y="{y}" class="{cls}"{anchor_attr}>{escape(value)}</text>'


def multiline(x: int, y: int, lines: list[str], cls: str, gap: int = 24) -> str:
    tspans = []
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else gap
        tspans.append(f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>')
    return f'<text x="{x}" y="{y}" class="{cls}">' + "".join(tspans) + "</text>"


def icon_chat(x: int, y: int, color: str) -> str:
    return f"""
      <path d="M{x} {y} h44 a10 10 0 0 1 10 10 v24 a10 10 0 0 1-10 10 h-20 l-12 11 v-11 h-12 a10 10 0 0 1-10-10 v-24 a10 10 0 0 1 10-10z" fill="none" stroke="{color}" stroke-width="4" stroke-linejoin="round"/>
      <circle cx="{x + 10}" cy="{y + 22}" r="3" fill="{color}"/><circle cx="{x + 22}" cy="{y + 22}" r="3" fill="{color}"/><circle cx="{x + 34}" cy="{y + 22}" r="3" fill="{color}"/>
    """


def icon_agent(x: int, y: int) -> str:
    return f"""
      <rect x="{x}" y="{y + 9}" width="58" height="45" rx="12" fill="#FFFFFF" stroke="#0072B2" stroke-width="3"/>
      <path d="M{x + 29} {y + 9}v-9m-6 0h12" stroke="#0072B2" stroke-width="3" stroke-linecap="round"/>
      <circle cx="{x + 19}" cy="{y + 30}" r="5" fill="#56B4E9"/><circle cx="{x + 39}" cy="{y + 30}" r="5" fill="#009E73"/>
      <path d="M{x + 18} {y + 43}q11 8 22 0" fill="none" stroke="#52616B" stroke-width="3" stroke-linecap="round"/>
    """


def conditional_generation_overlay() -> str:
    """Replace Stage 1 with frozen SymmCD and particle-guided SMC sampling."""
    return """
  <g id="conditional-generation-overlay">
    <rect x="43" y="39" width="468" height="1313" rx="27" fill="#F5FAFD" stroke="#B9DCEB" stroke-width="2"/>
    <circle cx="88" cy="90" r="28" fill="#0072B2"/>
    <text x="88" y="100" text-anchor="middle" class="stage-no">1</text>
    <text x="130" y="82" class="stage-title">Crystal generation</text>
    <text x="130" y="108" class="stage-sub">particle-guided, symmetry-constrained</text>

    <rect x="76" y="145" width="402" height="166" rx="18" fill="#FFFFFF" stroke="#83C2DF" stroke-width="2" filter="url(#shadow)"/>
    <text x="100" y="178" class="card-title">Multi-objective request</text>
    <rect x="100" y="194" width="105" height="53" rx="11" fill="#E1F1F8"/>
    <text x="112" y="215" class="tiny">Formula</text><text x="112" y="237" class="mono">BiTe</text>
    <rect x="215" y="194" width="100" height="53" rx="11" fill="#E1F1F8"/>
    <text x="227" y="215" class="tiny">Space group</text><text x="227" y="237" class="mono">194</text>
    <rect x="325" y="194" width="129" height="53" rx="11" fill="#E9F7F2" stroke="#75C7AA"/>
    <text x="337" y="215" class="tiny">Target particle</text><text x="337" y="237" class="mono">DP</text>
    <text x="100" y="274" class="small">Agent validates formula, SG, particle type and count</text>
    <rect x="100" y="284" width="354" height="18" rx="9" fill="#E7F4FA"/>
    <text x="277" y="298" text-anchor="middle" class="tiny">Generate 64 DP-enriched structures in SG 194</text>

    <path d="M277 317V337" class="arrow-blue"/>
    <rect x="76" y="345" width="402" height="225" rx="19" fill="#FFFFFF" stroke="#56B4E9" stroke-width="2.5" filter="url(#shadow)"/>
    <rect x="76" y="345" width="402" height="49" rx="19" fill="#D9EEF8"/><rect x="76" y="376" width="402" height="18" fill="#D9EEF8"/>
    <text x="100" y="378" class="card-title">Frozen SymmCD</text>
    <rect x="350" y="356" width="104" height="25" rx="12.5" fill="#0072B2"/>
    <text x="402" y="374" text-anchor="middle" class="tiny" style="fill:#FFFFFF">epoch699 fixed</text>
    <text x="100" y="426" class="label">64 parallel reverse-diffusion particles</text>
    <text x="100" y="452" class="small">atom types locked to formula; SG condition retained</text>
    <g transform="translate(100,470)">
      <path d="M0 22 C42 -5,76 50,118 22 S194 -5,236 22 S310 50,354 22" fill="none" stroke="#B9DCEB" stroke-width="3"/>
      <path d="M0 42 C42 12,76 69,118 42 S194 12,236 42 S310 69,354 42" fill="none" stroke="#56B4E9" stroke-width="3"/>
      <path d="M0 62 C42 35,76 88,118 62 S194 35,236 62 S310 88,354 62" fill="none" stroke="#0072B2" stroke-width="3"/>
      <circle cx="0" cy="22" r="5" fill="#B9DCEB"/><circle cx="354" cy="22" r="5" fill="#B9DCEB"/>
      <circle cx="0" cy="42" r="5" fill="#56B4E9"/><circle cx="354" cy="42" r="5" fill="#56B4E9"/>
      <circle cx="0" cy="62" r="5" fill="#0072B2"/><circle cx="354" cy="62" r="5" fill="#0072B2"/>
    </g>
    <text x="277" y="555" text-anchor="middle" class="tiny">asymmetric-unit coordinates + lattice + site symmetry</text>

    <path d="M277 577V597" class="arrow-blue"/>
    <rect x="76" y="605" width="402" height="174" rx="18" fill="#FFFFFF" stroke="#75C7AA" stroke-width="2" filter="url(#shadow)"/>
    <circle cx="111" cy="645" r="22" fill="#D7F1E8" stroke="#009E73" stroke-width="2"/>
    <text x="111" y="652" text-anchor="middle" class="label" fill="#007A59">DP</text>
    <text x="145" y="638" class="card-title">Particle surrogate</text>
    <text x="145" y="662" class="small">97 positives + 137 SG-compatible hard negatives</text>
    <rect x="100" y="688" width="164" height="35" rx="17.5" fill="#E9F7F2"/>
    <text x="182" y="711" text-anchor="middle" class="tiny">OOF PR-AUC 0.678</text>
    <rect x="274" y="688" width="180" height="35" rx="17.5" fill="#E9F7F2"/>
    <text x="364" y="711" text-anchor="middle" class="tiny">EF@20% = 1.745</text>
    <text x="100" y="751" class="small">Scores projected structures: P(DP | candidate)</text>
    <text x="100" y="770" class="tiny">Group-held-out gate passed; SG194 EF = 1.956</text>

    <path d="M277 786V806" class="arrow-blue"/>
    <rect x="76" y="814" width="402" height="224" rx="19" fill="#FFFFFF" stroke="#0072B2" stroke-width="2.5" filter="url(#shadow)"/>
    <text x="100" y="848" class="card-title">TDS-inspired SMC twisting</text>
    <text x="100" y="874" class="small">Gradual guidance over the final 50% of 500 steps</text>
    <g transform="translate(100,895)">
      <line x1="0" y1="24" x2="354" y2="24" stroke="#B9DCEB" stroke-width="8" stroke-linecap="round"/>
      <line x1="177" y1="24" x2="354" y2="24" stroke="#0072B2" stroke-width="8" stroke-linecap="round"/>
      <circle cx="0" cy="24" r="8" fill="#B9DCEB"/><circle cx="177" cy="24" r="8" fill="#56B4E9"/><circle cx="354" cy="24" r="8" fill="#0072B2"/>
      <text x="0" y="53" text-anchor="middle" class="tiny">beta 0</text><text x="177" y="53" text-anchor="middle" class="tiny">start scoring</text><text x="354" y="53" text-anchor="middle" class="tiny">beta 3</text>
    </g>
    <rect x="100" y="964" width="166" height="48" rx="12" fill="#E7F4FA"/>
    <text x="183" y="984" text-anchor="middle" class="tiny">score every 25 steps</text><text x="183" y="1003" text-anchor="middle" class="tiny">incremental weights</text>
    <path d="M267 988H286" class="arrow-blue"/>
    <rect x="296" y="964" width="158" height="48" rx="12" fill="#FFF3DF" stroke="#E69F00"/>
    <text x="375" y="984" text-anchor="middle" class="tiny">ESS &lt; 0.8 N</text><text x="375" y="1003" text-anchor="middle" class="tiny">systematic resample</text>
    <text x="277" y="1029" text-anchor="middle" class="tiny">copy coordinates, lattice, atom and site-symmetry states</text>

    <path d="M277 1045V1065" class="arrow-blue"/>
    <rect x="76" y="1073" width="402" height="190" rx="18" fill="#FFFFFF" stroke="#83C2DF" stroke-width="2" filter="url(#shadow)"/>
    <text x="100" y="1107" class="card-title">Condition-valid candidate pool</text>
    <g transform="translate(101,1125)">
      <circle cx="8" cy="8" r="8" fill="#D7F1E8"/><path d="M4 8l3 3 6-7" fill="none" stroke="#009E73" stroke-width="2"/><text x="26" y="13" class="small">composition retained</text>
      <circle cx="8" cy="38" r="8" fill="#D7F1E8"/><path d="M4 38l3 3 6-7" fill="none" stroke="#009E73" stroke-width="2"/><text x="26" y="43" class="small">actual SG = requested SG</text>
      <circle cx="8" cy="68" r="8" fill="#D7F1E8"/><path d="M4 68l3 3 6-7" fill="none" stroke="#009E73" stroke-width="2"/><text x="26" y="73" class="small">valid geometry + ranked DP proxy</text>
    </g>
    <rect x="100" y="1214" width="354" height="35" rx="17.5" fill="#0072B2"/>
    <text x="277" y="1237" text-anchor="middle" class="small" fill="#FFFFFF">CIF / POSCAR + probabilities + SMC audit trail</text>

    <rect x="76" y="1281" width="402" height="48" rx="13" fill="#FFF3DF" stroke="#E69F00"/>
    <text x="277" y="1302" text-anchor="middle" class="small">Proxy enrichment is not topology confirmation</text>
    <text x="277" y="1321" text-anchor="middle" class="tiny">SOC DFT + IRVSP remain required downstream</text>
    <path d="M478 1168H520V684H531" class="arrow"/>
  </g>
"""


def integrated_svg(original_body: str) -> str:
    cards = [
        (42, 470, "#0072B2", "01  GENERATE", "Frozen SymmCD + SMC", "formula / SG / DP-DNL / count"),
        (540, 430, "#009E73", "02  RELAX + ENERGY", "MACE", "CIF/POSCAR · Eform · actual SG"),
        (998, 690, "#D97706", "03  CALCULATE BANDS", "VASP / atomate2 / IRVSP", "DFT · SOC · band image"),
        (1716, 642, "#A64D9B", "04  RETRIEVE + EXPLAIN", "Local encyclopedia index", "particles · type · high-symmetry path"),
    ]
    card_markup = []
    for x, width, color, kicker, title_value, subtitle in cards:
        card_markup.append(
            f"""
    <g filter="url(#agent-shadow)">
      <rect x="{x}" y="326" width="{width}" height="116" rx="22" fill="#FFFFFF" stroke="{color}" stroke-width="2.5"/>
      <rect x="{x}" y="326" width="9" height="116" rx="4.5" fill="{color}"/>
    </g>
    {text(x + 30, 355, kicker, 'a-chip', None).replace('>', f' fill="{color}">', 1)}
    {text(x + 30, 389, title_value, 'a-card-title')}
    {text(x + 30, 417, subtitle, 'a-small')}
    <path d="M{x + width // 2} 442V492" class="a-dash" stroke="{color}"/>
            """
        )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:svg="http://www.w3.org/2000/svg"
     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:cc="http://creativecommons.org/ns#"
     xmlns:dc="http://purl.org/dc/elements/1.1/"
     width="2400" height="1900" viewBox="0 0 2400 1900" role="img" aria-labelledby="agent-integrated-title agent-integrated-desc">
  <title id="agent-integrated-title">SymmBand-Agent integrated topological-material discovery workflow</title>
  <desc id="agent-integrated-desc">A conversational Pydantic AI orchestration layer controls frozen-checkpoint particle-guided SymmCD generation with TDS-inspired sequential Monte Carlo, MACE energy and relaxation, electronic band calculations, and local emergent-particle knowledge retrieval.</desc>
{COMMON_DEFS}
  <rect width="2400" height="1900" fill="#FFFFFF"/>
  <rect width="2400" height="492" fill="#F7FBFD"/>
  <rect width="2400" height="492" fill="url(#agent-dots)"/>
  <rect x="0" y="488" width="2400" height="4" fill="#D8E7ED"/>

  {text(42, 39, 'CONVERSATIONAL AGENT ORCHESTRATION', 'a-kicker')}
  {text(42, 76, 'Natural-language intent becomes a validated, traceable scientific workflow', 'a-title')}

  <g filter="url(#agent-shadow)">
    <rect x="42" y="105" width="540" height="158" rx="25" fill="#FFFFFF" stroke="#B9DCEB" stroke-width="2"/>
  </g>
  {icon_chat(78, 139, '#0072B2')}
  {text(157, 142, 'USER REQUEST', 'a-kicker')}
  {multiline(157, 177, ['“Generate 10 NaBi structures in SG 194,', 'then calculate their bands.”'], 'a-mono', 27)}
  {text(157, 239, 'Chinese or English · interactive follow-up', 'a-small')}

  <path d="M582 184H635" class="a-arrow"/>
  <g filter="url(#agent-shadow)">
    <rect x="650" y="92" width="1100" height="184" rx="28" fill="url(#agent-core)" stroke="#8CC8D8" stroke-width="2.5"/>
  </g>
  {icon_agent(692, 123)}
  {text(778, 132, 'SymmBand-Agent', 'a-title')}
  {text(778, 162, 'Pydantic AI + DeepSeek API', 'a-subtitle')}
  <rect x="692" y="192" width="234" height="47" rx="23.5" fill="#E7F4FA" stroke="#56B4E9"/>
  {text(809, 222, 'intent extraction', 'a-chip', 'middle')}
  <rect x="945" y="192" width="250" height="47" rx="23.5" fill="#E8F6F1" stroke="#75C7AA"/>
  {text(1070, 222, 'parameter validation', 'a-chip', 'middle')}
  <rect x="1214" y="192" width="222" height="47" rx="23.5" fill="#FFF3DF" stroke="#E69F00"/>
  {text(1325, 222, 'tool routing', 'a-chip', 'middle')}
  <rect x="1455" y="192" width="245" height="47" rx="23.5" fill="#F7ECF5" stroke="#CC79A7"/>
  {text(1578, 222, 'session memory', 'a-chip', 'middle')}

  <path d="M1750 184H1803" class="a-arrow"/>
  <g filter="url(#agent-shadow)">
    <rect x="1818" y="105" width="540" height="158" rx="25" fill="#FFFFFF" stroke="#B8E0D3" stroke-width="2"/>
  </g>
  <circle cx="1868" cy="158" r="27" fill="#E8F6F1" stroke="#009E73" stroke-width="2"/>
  <path d="M1855 158l9 9 18-22" fill="none" stroke="#009E73" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
  {text(1910, 142, 'LIVE, STRUCTURED RESPONSE', 'a-kicker')}
  {multiline(1910, 177, ['progress · energy · actual SG', 'files · plots · sourced answers'], 'a-body', 28)}
  {text(1910, 239, 'Follow-up: “show the third structure”', 'a-small')}

  <path d="M1200 276V299H277V326M1200 299H755V326M1200 299H1343V326M1200 299H2037V326" class="a-dash"/>
  {''.join(card_markup)}

  <g transform="translate(0,500)">
{original_body}
{conditional_generation_overlay()}
  </g>
</svg>
"""


def capability_card(x: int, y: int, width: int, color: str, number: str, title_value: str,
                    system: str, lines: list[str], output: str) -> str:
    body = multiline(x + 34, y + 150, lines, "a-body", 27)
    return f"""
  <g filter="url(#agent-shadow)">
    <rect x="{x}" y="{y}" width="{width}" height="296" rx="28" fill="#FFFFFF" stroke="{color}" stroke-width="2.5"/>
  </g>
  <circle cx="{x + 52}" cy="{y + 53}" r="29" fill="{color}"/>
  {text(x + 52, y + 63, number, 'a-card-title', 'middle').replace('>', ' fill="#FFFFFF">', 1)}
  {text(x + 94, y + 48, title_value, 'a-card-title')}
  {text(x + 94, y + 75, system, 'a-small')}
  <line x1="{x + 34}" y1="{y + 105}" x2="{x + width - 34}" y2="{y + 105}" stroke="#DCE7EB" stroke-width="2"/>
  {body}
  <rect x="{x + 34}" y="{y + 244}" width="{width - 68}" height="34" rx="17" fill="{color}" opacity="0.11"/>
  {text(x + width // 2, y + 267, output, 'a-chip', 'middle').replace('>', f' fill="{color}">', 1)}
"""


def standalone_svg() -> str:
    cards = "".join([
        capability_card(70, 596, 430, "#0072B2", "1", "Crystal generation", "SymmCD workflow", ["Formula · space group · count", "Deduplication + actual SG", "Per-sample live progress"], "OUTPUT  POSCAR / CIF candidates"),
        capability_card(530, 596, 430, "#009E73", "2", "Energy + relaxation", "MACE potential", ["Read CIF or POSCAR", "Total / per-atom energy", "Formation energy + relaxed SG"], "OUTPUT  energies + structures"),
        capability_card(990, 596, 430, "#D97706", "3", "Electronic bands", "VASP · atomate2 · IRVSP", ["DFT and SOC workflows", "Symmetry representation", "Automated band plotting"], "OUTPUT  band_*.png + data"),
        capability_card(1450, 596, 430, "#A64D9B", "4", "Knowledge retrieval", "Local encyclopedia index", ["SOC / non-SOC particles", "Essential vs accidental", "High-symmetry k-paths"], "OUTPUT  sourced table + paths"),
    ])
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1950" height="1180" viewBox="0 0 1950 1180" role="img" aria-labelledby="agent-title agent-desc">
  <title id="agent-title">SymmBand-Agent capabilities</title>
  <desc id="agent-desc">A standalone diagram showing conversational requests routed through a Pydantic AI agent to crystal generation, MACE energy and relaxation, band calculation, and local scientific knowledge retrieval tools.</desc>
{COMMON_DEFS}
  <rect width="1950" height="1180" fill="#F7FBFD"/>
  <rect width="1950" height="1180" fill="url(#agent-dots)"/>
  {text(70, 58, 'SYMMBAND-AGENT', 'a-kicker')}
  {text(70, 105, 'Conversational control for symmetry-guided materials discovery', 'a-title')}
  {text(70, 139, 'One dialogue connects generation, atomistic energy, electronic bands, and curated symmetry knowledge.', 'a-subtitle')}

  <g filter="url(#agent-shadow)">
    <rect x="70" y="190" width="485" height="258" rx="28" fill="#FFFFFF" stroke="#B9DCEB" stroke-width="2"/>
  </g>
  {icon_chat(106, 227, '#0072B2')}
  {text(186, 230, 'CONVERSATION', 'a-kicker')}
  {multiline(106, 288, ['“Generate 10 BN structures in SG 194.”', '“What is the energy of graphene.cif?”', '“List accidental SOC particles in SG 216.”'], 'a-mono', 42)}
  <rect x="106" y="402" width="413" height="30" rx="15" fill="#E7F4FA"/>
  {text(312, 423, 'follow-up context: “show structure 3”', 'a-small', 'middle')}

  <path d="M555 319H630" class="a-arrow"/>
  <g filter="url(#agent-shadow)">
    <rect x="645" y="181" width="660" height="276" rx="34" fill="url(#agent-core)" stroke="#8CC8D8" stroke-width="3"/>
  </g>
  {icon_agent(700, 220)}
  {text(790, 230, 'Pydantic AI Agent', 'a-title')}
  {text(790, 262, 'DeepSeek API · typed tools · validated arguments', 'a-subtitle')}
  <line x1="700" y1="292" x2="1250" y2="292" stroke="#C7DDE5" stroke-width="2"/>
  <rect x="700" y="322" width="164" height="50" rx="25" fill="#E7F4FA" stroke="#56B4E9"/>
  {text(782, 354, 'understand', 'a-chip', 'middle')}
  <path d="M864 347H895" class="a-arrow"/>
  <rect x="910" y="322" width="164" height="50" rx="25" fill="#E8F6F1" stroke="#75C7AA"/>
  {text(992, 354, 'validate', 'a-chip', 'middle')}
  <path d="M1074 347H1105" class="a-arrow"/>
  <rect x="1120" y="322" width="130" height="50" rx="25" fill="#FFF3DF" stroke="#E69F00"/>
  {text(1185, 354, 'route', 'a-chip', 'middle')}
  <rect x="700" y="397" width="550" height="38" rx="19" fill="#F7ECF5"/>
  {text(975, 422, 'session memory · file registry · structured results', 'a-small', 'middle')}

  <path d="M1305 319H1380" class="a-arrow"/>
  <g filter="url(#agent-shadow)">
    <rect x="1395" y="190" width="485" height="258" rx="28" fill="#FFFFFF" stroke="#B8E0D3" stroke-width="2"/>
  </g>
  <circle cx="1443" cy="235" r="25" fill="#009E73"/>
  <path d="M1431 235l8 8 17-20" fill="none" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
  {text(1484, 230, 'TRACEABLE RESPONSE', 'a-kicker')}
  {multiline(1431, 289, ['Live logs and progress', 'Energy, formation energy, actual SG', 'Generated files and band images', 'Source-aware encyclopedia answers'], 'a-body', 34)}

  <path d="M975 457V522H285V596M975 522H745V596M975 522H1205V596M975 522H1665V596" class="a-dash"/>
  <rect x="790" y="493" width="370" height="45" rx="22.5" fill="#263238"/>
  {text(975, 523, 'TOOL ROUTER', 'a-chip', 'middle').replace('>', ' fill="#FFFFFF">', 1)}

{cards}

  <g filter="url(#agent-shadow)">
    <rect x="70" y="943" width="1810" height="155" rx="28" fill="#263238"/>
  </g>
  <text x="112" y="983" class="a-kicker a-on-dark-accent">EXAMPLE END-TO-END RUN</text>
  <text x="112" y="1025" class="a-body a-on-dark">Generate 10 NaBi candidates in SG 194</text>
  <path d="M489 1019H544" fill="none" stroke="#56B4E9" stroke-width="3" marker-end="url(#agent-arrow)"/>
  <text x="570" y="1025" class="a-body a-on-dark">relax + rank with MACE</text>
  <path d="M824 1019H879" fill="none" stroke="#75C7AA" stroke-width="3" marker-end="url(#agent-arrow)"/>
  <text x="905" y="1025" class="a-body a-on-dark">launch SOC band workflow</text>
  <path d="M1175 1019H1230" fill="none" stroke="#E69F00" stroke-width="3" marker-end="url(#agent-arrow)"/>
  <text x="1256" y="1025" class="a-body a-on-dark">save plots + structured report</text>
  <text x="112" y="1065" class="a-small a-on-dark-sub">Every tool call is explicit, inspectable, and reusable through conversational follow-up.</text>
  {text(1880, 1144, 'SymmBand-Agent · capability overview', 'a-small', 'end')}
</svg>
"""


def extract_original_body(source: str) -> str:
    start = source.find("<defs")
    end = source.rfind("</svg>")
    if start == -1 or end == -1:
        raise ValueError(f"Could not locate SVG body in {SOURCE}")
    body = source[start:end].rstrip()
    # Prevent duplicate accessibility IDs in the combined document.
    body = re.sub(r'\bid="svg177"', 'id="original-svg177"', body)
    return body


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    original_body = extract_original_body(source)
    INTEGRATED.write_text(integrated_svg(original_body), encoding="utf-8")
    STANDALONE.write_text(standalone_svg(), encoding="utf-8")
    print(f"Wrote {INTEGRATED.name}")
    print(f"Wrote {STANDALONE.name}")


if __name__ == "__main__":
    main()
