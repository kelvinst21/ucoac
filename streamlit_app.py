import re
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title='Bank Financial Dashboard',
    page_icon=':bank:',
    layout='wide',
)

KEYWORD_MAP = {
    'assets': ['activo', 'assets', 'asset', 'activos'],
    'liabilities': ['pasivo', 'liabilities', 'pasivos'],
    'equity': ['patrimonio', 'equity', 'capital', 'reserva'],
    'income': ['ingreso', 'income', 'revenue', 'ingresos', 'ventas', 'net_income', 'netincome'],
    'expense': ['gasto', 'expense', 'cost', 'costo', 'gastos', 'operating_cost'],
    'profit': ['utilidad', 'profit', 'ganancia', 'resultado', 'net_profit', 'netincome'],
    'portfolio': ['cartera', 'loan', 'portfolio', 'credito', 'credit', 'loan_portfolio', 'cartera_credito'],
    'liquidity': ['liquidez', 'liquidity', 'cash', 'caja', 'liquid'],
    'deposits': ['captacion', 'deposit', 'deposits', 'deposito', 'funding'],
    'delinquency': ['morosidad', 'npl', 'nonperforming', 'deterioro', 'delinquency'],
    'date': ['fecha', 'date', 'period', 'year', 'month', 'trim', 'quarter', 'periodo'],
    'coverage': ['cobertura', 'coverage', 'provision', 'reserva']
}

SCORE_COLUMNS = ['assets', 'liabilities', 'equity', 'income', 'expense', 'profit', 'portfolio', 'liquidity', 'deposits', 'delinquency', 'coverage']


def normalize_name(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r'[\s\-/\\]+', '_', value)
    value = re.sub(r'[^a-z0-9_]', '', value)
    value = re.sub(r'__+', '_', value)
    return value


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(r'[%$€\s]', '', regex=True)
        .str.replace(r',(?=\d{3}(?:\D|$))', '', regex=True)
        .str.replace(r'\.(?=\d{3}(?:\D|$))', '', regex=True)
        .str.replace(r'\s+', '', regex=True)
        .replace(['', 'nan', 'none', 'null', 'na'], np.nan),
        errors='coerce'
    )


def parse_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors='coerce')
    if parsed.notna().sum() > 0:
        return parsed
    extracted = series.astype(str).str.extract(r'([0-9]{4})')[0]
    return pd.to_datetime(extracted, format='%Y', errors='coerce')


def score_columns(columns: list[str], keywords: list[str]) -> list[tuple[int, str]]:
    scores = []
    for col in columns:
        normalized = col.lower()
        score = sum(1 for kw in keywords if kw in normalized)
        if score > 0:
            scores.append((score, col))
    return sorted(scores, key=lambda x: (-x[0], x[1]))


def identify_financial_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    categories = {}
    for category, keywords in KEYWORD_MAP.items():
        categories[category] = [col for _, col in score_columns(df.columns.tolist(), keywords)]
    return categories


def detect_date_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if any(kw in col for kw in KEYWORD_MAP['date']):
            parsed = parse_dates(df[col])
            if parsed.notna().any():
                return col
    for col in df.columns:
        if df[col].astype(str).str.match(r'^[0-9]{4}$').any():
            return col
    return None


from io import BytesIO, StringIO
import xml.etree.ElementTree as ET


def _strip_namespace(tag: str) -> str:
    return tag.split('}', 1)[-1] if '}' in tag else tag


def create_row(ruc, fecha, e_c, e_n, g_c, g_n, c_c, c_n, s_c, s_n, total):
    return {
        'RUC': ruc,
        'Fecha Corte': fecha,
        'elemento.codigo': e_c,
        'elemento.nombre': e_n,
        'grupo.codigo': g_c,
        'grupo.nombre': g_n,
        'cuenta.codigo': c_c,
        'cuenta.nombre': c_n or g_n,
        'subcuenta.codigo': s_c,
        'subcuenta.nombre': s_n,
        'total': total,
    }


def parse_balance_xml(file) -> pd.DataFrame:
    if hasattr(file, 'seek'):
        file.seek(0)
    tree = ET.parse(file)
    root = tree.getroot()
    namespace = ''
    ns = {}
    if '}' in root.tag:
        namespace = root.tag[root.tag.find('{') + 1:root.tag.find('}')]
        ns = {'ns': namespace}

    def qpath(tag: str) -> str:
        return f'.//ns:{tag}' if ns else f'.//{tag}'

    ruc = (
        root.get('rucEntidad')
        or root.get('ruc')
        or root.get('RUC')
        or root.get('nit')
        or ''
    )
    fecha_corte_raw = (
        root.get('fechaCorte')
        or root.get('fecha_corte')
        or root.get('fecha')
        or root.get('Fecha')
        or root.get('FechaCorte')
        or ''
    )
    fecha_corte = fecha_corte_raw
    if fecha_corte_raw:
        try:
            fecha_corte = datetime.strptime(fecha_corte_raw, '%d/%m/%Y').strftime('%Y-%m-%d')
        except ValueError:
            fecha_corte = fecha_corte_raw

    data_rows = []
    for elemento in root.findall(qpath('elemento'), ns):
        e_cod = elemento.get('codigo', '0')
        e_nom = elemento.get('nombre', 'N/A')

        for grupo in elemento.findall(qpath('grupo'), ns):
            g_cod = grupo.get('codigo', '0')
            g_nom = grupo.get('nombre', 'N/A')

            for cuenta in grupo.findall(qpath('cuenta'), ns):
                c_cod = cuenta.get('codigo', '0')
                c_nom = cuenta.get('nombre', 'N/A')

                subcuentas = cuenta.findall(qpath('subcuenta'), ns)
                if not subcuentas:
                    row = create_row(ruc, fecha_corte, e_cod, e_nom, g_cod, g_nom, c_cod, c_nom, '0', 'N/A', 0.0)
                    data_rows.append(row)
                else:
                    for sub in subcuentas:
                        s_cod = sub.get('codigo', '0')
                        s_nom = sub.get('nombre', 'N/A')
                        try:
                            total_val = float(sub.get('total', 0))
                        except (ValueError, TypeError):
                            total_val = 0.0

                        row = create_row(
                            ruc,
                            fecha_corte,
                            e_cod,
                            e_nom,
                            g_cod,
                            g_nom,
                            c_cod,
                            c_nom,
                            s_cod,
                            s_nom,
                            total_val,
                        )
                        data_rows.append(row)

    return pd.DataFrame(data_rows)


def extract_balance_xml_header(file) -> dict[str, str]:
    if hasattr(file, 'seek'):
        file.seek(0)
    tree = ET.parse(file)
    root = tree.getroot()
    namespace = ''
    ns = {}
    if '}' in root.tag:
        namespace = root.tag[root.tag.find('{') + 1:root.tag.find('}')]
        ns = {'ns': namespace}

    header = {}
    for key, value in root.attrib.items():
        if key.startswith('{'):
            continue
        header[normalize_name(_strip_namespace(key))] = value

    def qpath(tag: str) -> str:
        return f'.//ns:{tag}' if ns else f'.//{tag}'

    record_count = 1
    for tag in ['elemento', 'grupo', 'cuenta', 'subcuenta']:
        record_count += len(root.findall(qpath(tag), ns))
    header['registro_calculado'] = str(record_count)
    header['valor_cuadre_calculado'] = str(sum(parse_number(node.get('total')) for node in root.findall('.//*')))

    for elemento in root.findall(qpath('elemento'), ns):
        code = elemento.get('codigo', '').strip()
        total = parse_number(elemento.get('total'))
        if code:
            header[f'elemento_{code}_total'] = str(total)

    return header


def normalize_xml_filename_header(filename: str, header: dict[str, str]) -> bool:
    if not filename or not header:
        return False
    stem = Path(filename).stem
    parts = stem.split('_')
    if len(parts) < 3:
        return False
    estructura = parts[0].lower()
    ruc = parts[-2]
    fecha_file = parts[-1].replace('-', '/')
    fecha_header = header.get('fechacorte', '').replace('-', '/')
    return (
        estructura == header.get('estructura', '').lower()
        and ruc == header.get('rucentidad')
        and fecha_file == fecha_header
    )


def parse_number(value: str | float | int | None) -> float:
    try:
        return float(str(value).replace(',', '').strip())
    except Exception:
        return 0.0


def validate_balance_sheet(df: pd.DataFrame, header: dict[str, str] | None, filename: str | None) -> dict[str, dict[str, object]]:
    validations: dict[str, dict[str, object]] = {}
    if header is None:
        return validations

    actual_rows = int(header.get('registro_calculado', len(df)))
    expected_rows = int(header.get('numregistro', -1)) if header.get('numregistro') else -1
    header_match = normalize_xml_filename_header(filename or '', header)
    validations['registro'] = {
        'ok': expected_rows > 0 and actual_rows == expected_rows,
        'calculado': actual_rows,
        'esperado': expected_rows,
        'mensaje': 'Número de registros coincide con cabecera.' if expected_rows > 0 and actual_rows == expected_rows else 'Número de registros no coincide con cabecera.'
    }
    validations['cabecera'] = {
        'ok': header_match,
        'archivo': Path(filename).name if filename else '',
        'estructura': header.get('estructura', ''),
        'rucentidad': header.get('rucentidad', ''),
        'fechacorte': header.get('fechacorte', ''),
        'mensaje': 'Los datos de cabecera coinciden con el nombre del archivo.' if header_match else 'Los datos de cabecera no coinciden con el nombre del archivo.'
    }

    valor_cuadre_calculado = parse_number(header.get('valor_cuadre_calculado'))
    valor_cuadre_declarado = parse_number(header.get('valorcuadre'))
    validations['valor_cuadre'] = {
        'ok': abs(valor_cuadre_calculado - valor_cuadre_declarado) <= 1e-2,
        'calculado': valor_cuadre_calculado,
        'declarado': valor_cuadre_declarado,
        'mensaje': 'Valor de cuadre coincide con la suma de todos los totales XML.' if abs(valor_cuadre_calculado - valor_cuadre_declarado) <= 1e-2 else 'Valor de cuadre no coincide con la suma de todos los totales XML.'
    }

    def sum_element(code: str) -> float:
        if 'elemento.codigo' not in df:
            return 0.0
        return float(df.loc[df['elemento.codigo'].astype(str) == code, 'total'].sum())

    elementos_totals = {
        '1': parse_number(header.get('elemento_1_total')),
        '2': parse_number(header.get('elemento_2_total')),
        '3': parse_number(header.get('elemento_3_total')),
        '4': parse_number(header.get('elemento_4_total')),
        '5': parse_number(header.get('elemento_5_total')),
    }
    activos = elementos_totals['1']
    pasivos = elementos_totals['2']
    patrimonio = elementos_totals['3']
    gastos = elementos_totals['4']
    ingresos = elementos_totals['5']
    lefthand = activos + gastos
    righthand = pasivos + patrimonio + ingresos
    validations['ecuacion_contable'] = {
        'ok': abs(lefthand - righthand) <= 1e-2,
        'activos': activos,
        'gastos': gastos,
        'pasivos': pasivos,
        'patrimonio': patrimonio,
        'ingresos': ingresos,
        'mensaje': 'Ecuación contable fundamental se cumple.' if abs(lefthand - righthand) <= 1e-2 else 'Ecuación contable fundamental no se cumple.'
    }

    return validations


def load_xml(file) -> pd.DataFrame:
    if hasattr(file, 'seek'):
        file.seek(0)
    try:
        return pd.read_xml(file)
    except Exception:
        if hasattr(file, 'seek'):
            file.seek(0)
        try:
            return pd.read_xml(file, xpath='//row')
        except Exception:
            if hasattr(file, 'seek'):
                file.seek(0)
            return parse_balance_xml(file)


def load_data(file) -> tuple[pd.DataFrame, dict[str, str] | None]:
    if hasattr(file, 'seek'):
        file.seek(0)
    file_name = getattr(file, 'name', '') or ''
    try:
        first_bytes = file.read(1024)
    except Exception:
        first_bytes = b''
    if hasattr(file, 'seek'):
        file.seek(0)

    xml_flag = file_name.lower().endswith('.xml')
    if not xml_flag:
        if isinstance(first_bytes, (bytes, bytearray)):
            snippet = first_bytes.decode('utf-8', errors='ignore').lstrip()
        else:
            snippet = str(first_bytes).lstrip()
        if snippet.startswith('<'):
            xml_flag = True

    if xml_flag:
        if hasattr(file, 'seek'):
            file.seek(0)
        df = load_xml(file)
        if hasattr(file, 'seek'):
            file.seek(0)
        header = extract_balance_xml_header(file)
        return df, header

    try:
        return pd.read_csv(file, sep=None, engine='python', dtype=str), None
    except Exception:
        if hasattr(file, 'seek'):
            file.seek(0)
        return pd.read_csv(file, dtype=str), None


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_name(c) for c in df.columns]
    return df


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine='openpyxl')
    buffer.seek(0)
    return buffer.read()


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        'column': df.columns,
        'dtype': df.dtypes.astype(str),
        'non_null': df.notna().sum().values,
        'nulls': df.isna().sum().values,
        'unique': [df[c].nunique(dropna=True) for c in df.columns]
    })
    summary['null_pct'] = (summary['nulls'] / len(df) * 100).round(1)
    return summary


def build_time_index(df: pd.DataFrame, date_column: str | None) -> pd.DataFrame:
    df = df.copy()
    if date_column:
        parsed = parse_dates(df[date_column])
        if parsed.notna().any():
            df['period'] = parsed
            return df
    period_cols = [col for col in df.columns if re.fullmatch(r'\d{4}', col)]
    if period_cols:
        melted = df.melt(id_vars=[c for c in df.columns if c not in period_cols], value_vars=period_cols, var_name='period', value_name='value')
        melted['period'] = pd.to_datetime(melted['period'], format='%Y', errors='coerce')
        return melted
    return df


def safe_first(df: pd.DataFrame, candidate_cols: list[str]) -> str | None:
    return candidate_cols[0] if candidate_cols else None


def compute_kpis(df: pd.DataFrame, categories: dict[str, list[str]], date_column: str | None) -> dict[str, dict[str, float | None]]:
    kpis = {}
    values = {}
    for label in SCORE_COLUMNS:
        col = safe_first(df, categories[label])
        values[label] = df[col] if col and col in df else None
    assets = values['assets']
    liabilities = values['liabilities']
    equity = values['equity']
    income = values['income']
    expense = values['expense']
    profit = values['profit']
    portfolio = values['portfolio']
    liquidity = values['liquidity']
    delinquency = values['delinquency']
    coverage = values['coverage']

    total_assets = float(assets.sum()) if assets is not None and assets.notna().any() else None
    total_liabilities = float(liabilities.sum()) if liabilities is not None and liabilities.notna().any() else None
    total_equity = float(equity.sum()) if equity is not None and equity.notna().any() else None
    total_income = float(income.sum()) if income is not None and income.notna().any() else None
    total_expense = float(expense.sum()) if expense is not None and expense.notna().any() else None
    total_profit = float(profit.sum()) if profit is not None and profit.notna().any() else None
    total_portfolio = float(portfolio.sum()) if portfolio is not None and portfolio.notna().any() else None
    total_liquidity = float(liquidity.sum()) if liquidity is not None and liquidity.notna().any() else None
    total_delinquency = float(delinquency.sum()) if delinquency is not None and delinquency.notna().any() else None
    total_coverage = float(coverage.sum()) if coverage is not None and coverage.notna().any() else None

    roa = (total_profit / total_assets * 100) if total_profit is not None and total_assets else None
    roe = (total_profit / total_equity * 100) if total_profit is not None and total_equity else None
    liquidity_ratio = None
    if total_liquidity is not None and total_assets:
        liquidity_ratio = total_liquidity / total_assets
    elif liquidity is not None and assets is not None and assets.notna().any():
        liquidity_ratio = float(liquidity.mean())
    npl_ratio = None
    if total_delinquency is not None and total_portfolio:
        npl_ratio = total_delinquency / total_portfolio * 100
    cost_income = None
    if total_expense is not None and total_income:
        cost_income = total_expense / total_income * 100
    margin = None
    if total_income is not None and total_assets:
        margin = (total_income - total_expense) / total_assets * 100 if total_expense is not None else None
    coverage_ratio = None
    if total_coverage is not None and total_delinquency:
        coverage_ratio = total_coverage / total_delinquency * 100

    kpis['totals'] = {
        'Total Activos': total_assets,
        'Total Pasivos': total_liabilities,
        'Patrimonio': total_equity,
        'Ingresos': total_income,
        'Gastos': total_expense,
        'Utilidad Neta': total_profit,
        'Cartera': total_portfolio,
        'Liquidez Reportada': total_liquidity,
        'Morosidad Reportada': total_delinquency,
    }
    kpis['ratios'] = {
        'ROA (%)': roa,
        'ROE (%)': roe,
        'Ratio Liquidez': liquidity_ratio,
        'Ratio Morosidad (%)': npl_ratio,
        'Cost to Income (%)': cost_income,
        'Margen Financiero (%)': margin,
        'Cobertura Cartera (%)': coverage_ratio,
    }
    return kpis


def build_insights(df: pd.DataFrame, kpis: dict[str, dict[str, float | None]], categories: dict[str, list[str]], date_column: str | None) -> list[str]:
    insights = []
    ratios = kpis['ratios']
    if ratios['ROA (%)'] is not None:
        insights.append(f'ROA estimado de {ratios["ROA (%)"]:.2f}% sugiere la eficacia en el uso de los activos.')
    if ratios['ROE (%)'] is not None:
        insights.append(f'ROE estimado de {ratios["ROE (%)"]:.2f}% indica retorno sobre el capital.')
    if ratios['Cost to Income (%)'] is not None and ratios['Cost to Income (%)'] > 60:
        insights.append('El Cost to Income es superior a 60%, lo que indica presión operativa relevante.')
    if ratios['Ratio Morosidad (%)'] is not None and ratios['Ratio Morosidad (%)'] > 5:
        insights.append('La morosidad supera el 5%, señal de riesgo en la calidad de cartera.')
    if ratios['Ratio Liquidez'] is not None and ratios['Ratio Liquidez'] < 0.2:
        insights.append('La liquidez relativa es baja en comparación con el total de activos.')

    if date_column and 'period' in df.columns:
        df_period = df.dropna(subset=['period']).copy()
        if not df_period.empty and categories['income']:
            inc_col = safe_first(df_period, categories['income'])
            if inc_col and inc_col in df_period:
                trend = df_period.sort_values('period').groupby(df_period['period'].dt.to_period('M'))[inc_col].sum()
                if len(trend) >= 2:
                    last, prior = trend.iloc[-1], trend.iloc[-2]
                    if prior != 0:
                        growth = (last - prior) / abs(prior) * 100
                        insights.append(f'Ingresos crecieron {growth:.1f}% en el último mes disponible.')

    if not insights:
        insights.append('No se detectaron alertas graves con los datos disponibles. Verifique la estructura del archivo o etiquetas de columnas.')

    return insights


def plot_gauge(value: float | None, title: str, min_value: float = 0, max_value: float = 100, unit: str = '%') -> go.Figure:
    if value is None or np.isnan(value):
        value = min_value
    return go.Figure(go.Indicator(
        mode='gauge+number',
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [min_value, max_value]},
            'bar': {'color': '#1f77b4'},
            'steps': [
                {'range': [min_value, max_value * 0.6], 'color': '#d62728'},
                {'range': [max_value * 0.6, max_value * 0.85], 'color': '#ff7f0e'},
                {'range': [max_value * 0.85, max_value], 'color': '#2ca02c'}
            ]
        },
        number={'suffix': unit}
    ))


def build_charts(df: pd.DataFrame, categories: dict[str, list[str]], date_column: str | None) -> dict[str, go.Figure]:
    charts = {}
    df_chart = df.copy()
    if date_column and 'period' in df_chart.columns:
        df_chart = df_chart.dropna(subset=['period']).copy()
    else:
        if 'period' in df_chart.columns:
            df_chart = df_chart.dropna(subset=['period']).copy()

    date_key = 'period' if 'period' in df_chart.columns else None

    income_col = safe_first(df_chart, categories['income'])
    expense_col = safe_first(df_chart, categories['expense'])
    profit_col = safe_first(df_chart, categories['profit'])
    delinquency_col = safe_first(df_chart, categories['delinquency'])
    portfolio_col = safe_first(df_chart, categories['portfolio'])
    liquidity_col = safe_first(df_chart, categories['liquidity'])

    if date_key and income_col:
        charts['income_trend'] = px.line(df_chart, x=date_key, y=income_col, title='Ingresos por periodo', markers=True)
    if date_key and expense_col:
        charts['expense_trend'] = px.line(df_chart, x=date_key, y=expense_col, title='Gastos por periodo', markers=True)
    if date_key and profit_col:
        charts['profit_trend'] = px.line(df_chart, x=date_key, y=profit_col, title='Utilidad Neta por periodo', markers=True)
    if date_key and delinquency_col and portfolio_col:
        df_chart['npl_ratio'] = df_chart[delinquency_col] / df_chart[portfolio_col] * 100
        charts['npl_trend'] = px.line(df_chart, x=date_key, y='npl_ratio', title='Ratio de Morosidad (%)', markers=True)
    if date_key and income_col and expense_col:
        charts['income_expense'] = px.bar(df_chart, x=date_key, y=[income_col, expense_col], title='Ingresos vs Gastos', barmode='group')
    if portfolio_col and len(df_chart) <= 50:
        labels = df_chart.index.astype(str)
        charts['portfolio_treemap'] = px.treemap(df_chart.assign(label=labels), path=['label'], values=portfolio_col, title='Cartera por fila')
    return charts


def show_summary(metrics: dict[str, dict[str, float | None]], categories: dict[str, list[str]], metadata: pd.DataFrame):
    st.markdown('### KPIs Ejecutivos')
    cols = st.columns(4)
    labels = ['Total Activos', 'Total Pasivos', 'Patrimonio', 'Utilidad Neta']
    for idx, label in enumerate(labels):
        value = metrics['totals'].get(label)
        cols[idx].metric(label, f'{value:,.0f}' if value is not None else 'N/A')

    cols = st.columns(4)
    ratios = ['ROA (%)', 'ROE (%)', 'Ratio Liquidez', 'Ratio Morosidad (%)']
    for idx, label in enumerate(ratios):
        value = metrics['ratios'].get(label)
        cols[idx].metric(label, f'{value:.2f}%' if value is not None else 'N/A')

    with st.expander('Variables detectadas y categoría de columnas'):
        mapped = [{'Categoria': category, 'Columnas detectadas': ', '.join(categories[category]) or 'No detectado'} for category in SCORE_COLUMNS]
        st.table(pd.DataFrame(mapped))

    with st.expander('Calidad de datos y estructura del dataset'):
        st.write(metadata)


def show_validation_results(validations: dict[str, dict[str, object]]):
    if not validations:
        return
    st.markdown('### Validaciones de integridad del balance')
    rows = []
    for key, result in validations.items():
        status = 'OK' if result.get('ok') else 'FALLO'
        message = result.get('mensaje', '')
        details = []
        if key == 'registro':
            details.append(f"Registros: {result.get('actual')} / {result.get('esperado')}")
        elif key == 'cabecera':
            details.append(f"Archivo: {result.get('archivo')}")
            details.append(f"Estructura: {result.get('estructura')}")
            details.append(f"RUC: {result.get('rucentidad')}")
            details.append(f"Fecha Corte: {result.get('fechacorte')}")
        elif key == 'valor_cuadre':
            details.append(f"Calculado: {result.get('calculado')}")
            details.append(f"Declarado: {result.get('declarado')}")
        elif key == 'ecuacion_contable':
            details.append(f"Activo + Gastos: {result.get('activos') + result.get('gastos')}")
            details.append(f"Pasivo + Patrimonio + Ingresos: {result.get('pasivos') + result.get('patrimonio') + result.get('ingresos')}")
        rows.append({'Validación': message, 'Estado': status, 'Detalles': ' | '.join(details)})
    st.table(pd.DataFrame(rows))


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    sample_path = Path(__file__).parent / 'data' / 'gdp_data.csv'
    return pd.read_csv(sample_path, dtype=str)


def main():
    st.title('Dashboard Ejecutivo Financiero Bancario')
    st.markdown(
        'Cargue un CSV con información financiera del banco para generar un dashboard ejecutivo con análisis automático, KPIs y visualizaciones interactivas.'
    )

    with st.sidebar:
        st.header('Carga de datos')
        uploaded_file = st.file_uploader('Sube tu archivo bancario (CSV o XML)', type=['csv', 'xml'])
        if uploaded_file is not None:
            st.success('Archivo cargado correctamente.')
        st.markdown('---')
        st.write('El sistema detectará columnas de activos, pasivos, patrimonio, ingresos, gastos, cartera, liquidez, morosidad y fechas.')

    if uploaded_file is None:
        st.warning('No se ha cargado un archivo bancario. Se usa el dataset de ejemplo de GDP solo para demostrar la estructura de la app.')
        raw_df, raw_header = load_sample_data(), None
        uploaded_filename = None
        raw_df_original = raw_df.copy()
    else:
        try:
            raw_df, raw_header = load_data(uploaded_file)
            uploaded_filename = uploaded_file.name
            raw_df_original = raw_df.copy()
        except Exception as err:
            st.error(f'Error al leer el archivo: {err}')
            return

    if raw_df.empty:
        st.error('El archivo no contiene datos o no se pudo leer. Verifique el formato del archivo.')
        return

    df = normalize_dataframe(raw_df)
    categories = identify_financial_columns(df)
    date_column = detect_date_column(df)
    if date_column:
        df[date_column] = parse_dates(df[date_column])
    df = df.apply(safe_numeric)
    df = build_time_index(df, date_column)
    metadata = summarize_dataframe(df)
    kpis = compute_kpis(df, categories, date_column)
    insights = build_insights(df, kpis, categories, date_column)
    charts = build_charts(df, categories, date_column)

    if uploaded_filename and uploaded_filename.lower().endswith('.xml'):
        try:
            excel_bytes = df_to_excel_bytes(raw_df_original)
            excel_name = Path(uploaded_filename).stem + '.xlsx'
            st.download_button(
                label='Descargar archivo convertido a Excel',
                data=excel_bytes,
                file_name=excel_name,
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        except Exception as err:
            st.error(f'No se pudo generar el Excel: {err}')

    validation_results = validate_balance_sheet(raw_df_original, raw_header, uploaded_filename)
    show_validation_results(validation_results)


if __name__ == '__main__':
    main()
