from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st


def _formatar_brl(valor: float) -> str:
    """Formata um número float para moeda brasileira (R$ 1.234,56)."""
    sinal = "-" if valor < 0 else ""
    abs_val = abs(valor)
    return f"{sinal}R$ {abs_val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _formatar_pct(valor: float | None) -> str:
    """Formata um percentual float com 1 casa decimal."""
    if valor is None or pd.isna(valor):
        return "Sem custo"
    return f"{valor:.1f}%".replace(".", ",")


def _obter_janela_datas(df_movs: pd.DataFrame | None, dias_periodo: int) -> tuple[pd.Timestamp, pd.Timestamp, list[str]]:
    """Define a data inicial, data final e a lista de datas YYYY-MM-DD da janela analisada."""
    dias_validos = max(int(dias_periodo), 1)
    data_fim = pd.Timestamp.now().normalize()

    if df_movs is not None and not df_movs.empty and "Data_Emissao" in df_movs.columns:
        datas_validas = pd.to_datetime(df_movs["Data_Emissao"], errors="coerce").dropna()
        if not datas_validas.empty:
            data_fim = datas_validas.max().normalize()
            data_min_xml = datas_validas.min().normalize()
            span_xml = int((data_fim - data_min_xml).days) + 1
            if span_xml > dias_validos:
                data_inicio = data_min_xml
            else:
                data_inicio = data_fim - pd.Timedelta(days=dias_validos - 1)
            datas_idx = pd.date_range(start=data_inicio, end=data_fim, freq="D")
            return data_inicio, data_fim, [d.strftime("%Y-%m-%d") for d in datas_idx]

    data_inicio = data_fim - pd.Timedelta(days=dias_validos - 1)
    datas_idx = pd.date_range(start=data_inicio, end=data_fim, freq="D")
    return data_inicio, data_fim, [d.strftime("%Y-%m-%d") for d in datas_idx]


def _reconstruir_kardex_diario(
    df_base: pd.DataFrame,
    df_movs: pd.DataFrame | None,
    dias_periodo: int,
    col_inicial: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Reconstrói o extrato diário (Kardex) de cada SKU dia a dia.
    Retorna:
      - df_metricas_hist: resumo histórico por SKU
      - df_kardex_completo: tabela dia a dia com Saldo_Abertura, Entradas, Saidas, Saldo_Fechamento e Em_Ruptura
      - data_fim: data final de referência da janela
    """
    _, data_fim, lista_datas = _obter_janela_datas(df_movs, dias_periodo)
    n_dias_janela = max(len(lista_datas), 1)

    mov_diario_map = {}
    if df_movs is not None and not df_movs.empty and {"SKU", "Tipo", "Data_Emissao", "Quantidade"}.issubset(df_movs.columns):
        df_m = df_movs.copy()
        df_m["SKU"] = df_m["SKU"].astype(str).str.strip()
        df_m["Data_Dia"] = df_m["Data_Emissao"].astype(str).str.slice(0, 10)
        df_m["Quantidade"] = pd.to_numeric(df_m["Quantidade"], errors="coerce").fillna(0.0)

        agrupado = df_m.groupby(["SKU", "Data_Dia", "Tipo"], as_index=False)["Quantidade"].sum()
        for row in agrupado.itertuples(index=False):
            chave = (row.SKU, row.Data_Dia)
            if chave not in mov_diario_map:
                mov_diario_map[chave] = {"Entrada": 0.0, "Saída": 0.0}
            if row.Tipo == "Entrada":
                mov_diario_map[chave]["Entrada"] += float(row.Quantidade)
            elif row.Tipo == "Saída":
                mov_diario_map[chave]["Saída"] += float(row.Quantidade)

    registros_hist = []
    registros_kardex = []

    for row in df_base.itertuples(index=False):
        sku = str(getattr(row, "SKU")).strip()
        desc = str(getattr(row, "Descrição", ""))
        cat = str(getattr(row, "Categoria", "Geral"))
        marca = str(getattr(row, "Marca", "Sem Marca"))
        saldo_corrente = float(getattr(row, col_inicial, 0.0) or 0.0)

        dias_com_estoque = 0
        dias_zerado = 0
        episodios_ruptura = 0
        soma_saldos_diarios = 0.0
        estava_zerado_antes = saldo_corrente <= 0

        for data_str in lista_datas:
            saldo_abertura = saldo_corrente
            mov = mov_diario_map.get((sku, data_str), {"Entrada": 0.0, "Saída": 0.0})
            entradas_dia = mov["Entrada"]
            saidas_dia = mov["Saída"]
            saldo_fechamento = saldo_abertura + entradas_dia - saidas_dia

            # Considera que houve disponibilidade no dia se começou com estoque ou recebeu mercadoria no dia
            disponivel_no_dia = (saldo_abertura > 0) or (entradas_dia > 0)
            em_ruptura_dia = saldo_fechamento <= 0

            if disponivel_no_dia:
                dias_com_estoque += 1
            if em_ruptura_dia:
                dias_zerado += 1
                if not estava_zerado_antes:
                    episodios_ruptura += 1

            estava_zerado_antes = em_ruptura_dia
            soma_saldos_diarios += max(saldo_fechamento, 0.0)
            saldo_corrente = saldo_fechamento

            registros_kardex.append(
                {
                    "SKU": sku,
                    "Descrição": desc,
                    "Categoria": cat,
                    "Marca": marca,
                    "Data": data_str,
                    "Saldo_Abertura": round(saldo_abertura, 2),
                    "Entradas": round(entradas_dia, 2),
                    "Saidas": round(saidas_dia, 2),
                    "Saldo_Fechamento": round(saldo_fechamento, 2),
                    "Em_Ruptura": em_ruptura_dia,
                    "Situacao_Dia": "🔴 Zerado / Ruptura" if em_ruptura_dia else "🟢 Com Estoque",
                }
            )

        disponibilidade_pct = round((dias_com_estoque / n_dias_janela) * 100, 1)
        estoque_medio_diario = round(soma_saldos_diarios / n_dias_janela, 2)

        registros_hist.append(
            {
                "SKU": sku,
                "Dias_Janela_Hist": n_dias_janela,
                "Dias_Com_Estoque": dias_com_estoque,
                "Dias_Ruptura_Hist": dias_zerado,
                "Episodios_Ruptura": episodios_ruptura,
                "Disponibilidade_Pct": disponibilidade_pct,
                "Estoque_Medio_Diario": estoque_medio_diario,
            }
        )

    df_metricas_hist = pd.DataFrame(registros_hist)
    df_kardex_completo = pd.DataFrame(registros_kardex)
    return df_metricas_hist, df_kardex_completo, data_fim


def _calcular_dias_sem_giro(df: pd.DataFrame, data_referencia: pd.Timestamp) -> pd.Series:
    """Calcula quantos dias se passaram desde a última saída registrada até a data de referência."""
    if "Ultima_Saida" not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)

    datas_saida = pd.to_datetime(df["Ultima_Saida"], errors="coerce")
    dias = (data_referencia - datas_saida).dt.days
    return dias.clip(lower=0)


def calcular_indicadores_estoque(
    df_consolidado: pd.DataFrame,
    df_movs: pd.DataFrame | None = None,
    dias_periodo: int = 30,
    cobertura_minima: int = 7,
    cobertura_alvo: int = 15,
    dias_excesso: int = 45,
    pct_deducoes_liq: float = 15.0,
    regras_categoria: dict[str, dict[str, int]] | None = None,
    snap_anterior: dict | None = None,
    usar_dmd_acumulada: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula os KPIs tendo a reconstrução histórica dia a dia (Kardex) como base principal,
    incluindo Margem Bruta (via XML) e Margem Líquida Estimada (via % deduções/despesas).
    """
    df = df_consolidado.copy()
    df["SKU"] = df["SKU"].astype(str).str.strip()

    if "Categoria" not in df.columns:
        df["Categoria"] = "Geral"
    else:
        df["Categoria"] = df["Categoria"].fillna("Geral").astype(str).str.strip().replace("", "Geral")

    if "Marca" not in df.columns:
        df["Marca"] = "Sem Marca"
    else:
        df["Marca"] = df["Marca"].fillna("Sem Marca").astype(str).str.strip().replace("", "Sem Marca")

    col_inicial = "Qtd Inicial" if "Qtd Inicial" in df.columns else "Qtd Física"
    if col_inicial not in df.columns:
        df[col_inicial] = 0.0

    # 1. Reconstrução histórica dia a dia (Passado do SKU)
    df_hist, df_kardex, data_ref = _reconstruir_kardex_diario(
        df_base=df,
        df_movs=df_movs,
        dias_periodo=dias_periodo,
        col_inicial=col_inicial,
    )
    df = df.merge(df_hist, on="SKU", how="left")

    # 2. Demanda Média Diária REAL (descontando os dias em que o SKU estava sem estoque)
    df["DMD_Bruta_Lote"] = (df["Total_Saidas"] / df["Dias_Janela_Hist"].clip(lower=1)).round(2)
    df["DMD_Real_Hist"] = np.where(
        df["Dias_Com_Estoque"] > 0,
        (df["Total_Saidas"] / df["Dias_Com_Estoque"]).round(2),
        df["DMD_Bruta_Lote"],
    )

    # 3. Histórico Acumulado entre ciclos (Ponto de Partida)
    saidas_prev = pd.Series(0.0, index=df.index)
    dias_disp_prev = pd.Series(0.0, index=df.index)
    if snap_anterior is not None and "df_base_proximo_ciclo" in snap_anterior:
        df_ref = snap_anterior["df_base_proximo_ciclo"].copy()
        if "SKU" in df_ref.columns:
            df_ref["SKU"] = df_ref["SKU"].astype(str).str.strip()
            if "Saidas_Acum_Ref" in df_ref.columns:
                mapa_s = dict(zip(df_ref["SKU"], pd.to_numeric(df_ref["Saidas_Acum_Ref"], errors="coerce").fillna(0.0)))
                saidas_prev = df["SKU"].map(mapa_s).fillna(0.0)
            if "Dias_Disp_Acum_Ref" in df_ref.columns:
                mapa_d = dict(zip(df_ref["SKU"], pd.to_numeric(df_ref["Dias_Disp_Acum_Ref"], errors="coerce").fillna(0.0)))
                dias_disp_prev = df["SKU"].map(mapa_d).fillna(0.0)
            elif "Dias_Acum_Ref" in df_ref.columns:
                mapa_d = dict(zip(df_ref["SKU"], pd.to_numeric(df_ref["Dias_Acum_Ref"], errors="coerce").fillna(0.0)))
                dias_disp_prev = df["SKU"].map(mapa_d).fillna(0.0)

    df["Saidas_Acumuladas"] = (saidas_prev + df["Total_Saidas"]).round(2)
    df["Dias_Disp_Acumulados"] = (dias_disp_prev + df["Dias_Com_Estoque"]).round(0)
    df["DMD_Real_Acumulada"] = np.where(
        df["Dias_Disp_Acumulados"] > 0,
        (df["Saidas_Acumuladas"] / df["Dias_Disp_Acumulados"]).round(2),
        df["DMD_Real_Hist"],
    )

    tem_historico_acum = bool((dias_disp_prev > 0).any())
    df["DMD_Referencia"] = np.where(
        usar_dmd_acumulada & tem_historico_acum,
        df["DMD_Real_Acumulada"],
        df["DMD_Real_Hist"],
    )

    # 4. Giro Real baseado no Estoque Médio Diário reconstruído
    df["Giro_Estoque"] = np.where(
        df["Estoque_Medio_Diario"] > 0,
        (df["Total_Saidas"] / df["Estoque_Medio_Diario"]).round(2),
        0.0,
    )

    # 5. Preço Médio de Venda, Custo Médio de Compra e Valor em Estoque
    df["Preco_Medio_Venda"] = np.where(
        df["Total_Saidas"] > 0,
        (df["Valor_Saidas"] / df["Total_Saidas"]).round(2),
        0.0,
    )
    df["Custo_Medio_Compra"] = np.where(
        df["Total_Entradas"] > 0,
        (df["Valor_Entradas"] / df["Total_Entradas"]).round(2),
        0.0,
    )
    valor_unitario_ref = np.where(
        df["Custo_Medio_Compra"] > 0,
        df["Custo_Medio_Compra"],
        df["Preco_Medio_Venda"],
    )
    df["Valor_Estoque_Atual"] = (df["Saldo_Calculado"].clip(lower=0) * valor_unitario_ref).round(2)

    # 6. Margem Bruta (via XML) e Margem Líquida Estimada (via % deduções/despesas)
    taxa_deducao = max(float(pct_deducoes_liq), 0.0) / 100.0
    df["Tem_Custo_XML"] = (df["Custo_Medio_Compra"] > 0) & (df["Valor_Saidas"] > 0)
    df["CMV_Vendido"] = np.where(
        df["Tem_Custo_XML"],
        (df["Total_Saidas"] * df["Custo_Medio_Compra"]).round(2),
        0.0,
    )
    df["Lucro_Bruto"] = np.where(
        df["Tem_Custo_XML"],
        (df["Valor_Saidas"] - df["CMV_Vendido"]).round(2),
        0.0,
    )
    df["Margem_Bruta_Pct"] = np.where(
        df["Tem_Custo_XML"],
        ((df["Lucro_Bruto"] / df["Valor_Saidas"]) * 100).round(1),
        np.nan,
    )
    df["Deducoes_Operacionais"] = np.where(
        df["Tem_Custo_XML"],
        (df["Valor_Saidas"] * taxa_deducao).round(2),
        0.0,
    )
    df["Lucro_Liquido_Est"] = np.where(
        df["Tem_Custo_XML"],
        (df["Lucro_Bruto"] - df["Deducoes_Operacionais"]).round(2),
        0.0,
    )
    df["Margem_Liquida_Pct"] = np.where(
        df["Tem_Custo_XML"],
        ((df["Lucro_Liquido_Est"] / df["Valor_Saidas"]) * 100).round(1),
        np.nan,
    )

    # 7. Impacto da Ruptura Passada (O que deixou de ser vendido nos dias zerados do período)
    df["Unidades_Perdidas_Hist"] = np.where(
        (df["Dias_Ruptura_Hist"] > 0) & (df["DMD_Real_Hist"] > 0),
        np.ceil(df["DMD_Real_Hist"] * df["Dias_Ruptura_Hist"]),
        0.0,
    )
    df["Perda_Financeira_Hist"] = (
        df["Unidades_Perdidas_Hist"] * df["Preco_Medio_Venda"]
    ).round(2)
    df["Lucro_Bruto_Perdido_Hist"] = np.where(
        df["Tem_Custo_XML"],
        (df["Unidades_Perdidas_Hist"] * (df["Preco_Medio_Venda"] - df["Custo_Medio_Compra"])).round(2),
        0.0,
    )
    df["Perda_Diaria_Ruptura"] = np.where(
        (df["Saldo_Calculado"] <= 0) & (df["DMD_Referencia"] > 0),
        (df["DMD_Referencia"] * df["Preco_Medio_Venda"]).round(2),
        0.0,
    )

    # 8. Curva ABC pela Receita Potencial (Faturamento Realizado + Venda Perdida por Ruptura)
    df["Receita_Potencial_Hist"] = (df["Valor_Saidas"] + df["Perda_Financeira_Hist"]).round(2)
    usar_financeiro = df["Receita_Potencial_Hist"].sum() > 0
    criterio_abc = "Receita_Potencial_Hist" if usar_financeiro else "Total_Saidas"

    df = df.sort_values(by=criterio_abc, ascending=False).reset_index(drop=True)
    total_geral = df[criterio_abc].sum()

    if total_geral > 0:
        df["Participacao_Pct"] = ((df[criterio_abc] / total_geral) * 100).round(2)
        df["Acumulado_Pct"] = df["Participacao_Pct"].cumsum().round(2)
        df["Curva_ABC"] = np.select(
            [df["Acumulado_Pct"] <= 80.0, df["Acumulado_Pct"] <= 95.0],
            ["A", "B"],
            default="C",
        )
    else:
        df["Participacao_Pct"] = 0.0
        df["Acumulado_Pct"] = 0.0
        df["Curva_ABC"] = "C"

    # 9. Projeção Futura: Cobertura (dias) e Data Estimada de Ruptura
    df["Cobertura_Dias"] = np.where(
        df["DMD_Referencia"] > 0,
        (df["Saldo_Calculado"].clip(lower=0) / df["DMD_Referencia"]).round(1),
        np.where(df["Saldo_Calculado"] > 0, 999.0, 0.0),
    )

    def _projetar_data_ruptura(row_item) -> str:
        if row_item["Saldo_Calculado"] <= 0:
            return "Já zerado"
        if row_item["DMD_Referencia"] <= 0 or row_item["Cobertura_Dias"] >= 999.0:
            return "Sem consumo"
        dias_restantes = int(np.floor(row_item["Cobertura_Dias"]))
        data_prev = data_ref + timedelta(days=dias_restantes)
        return data_prev.strftime("%d/%m/%Y")

    df["Previsao_Ruptura"] = df.apply(_projetar_data_ruptura, axis=1)
    df["Dias_Sem_Giro"] = _calcular_dias_sem_giro(df, data_ref)

    # 10. Cobertura Mínima e Cobertura Alvo aplicadas por Categoria
    if regras_categoria:
        df["Cob_Min_Aplicada"] = df["Categoria"].map(
            lambda c: int(regras_categoria.get(c, {}).get("min", cobertura_minima))
        )
        df["Cob_Alvo_Aplicada"] = df["Categoria"].map(
            lambda c: int(regras_categoria.get(c, {}).get("alvo", cobertura_alvo))
        )
    else:
        df["Cob_Min_Aplicada"] = int(cobertura_minima)
        df["Cob_Alvo_Aplicada"] = int(cobertura_alvo)

    # 11. Diagnóstico Futuro + Sinalização de Ruptura Oculta no Passado
    df["Status_Estoque"] = np.select(
        [
            df["Saldo_Calculado"] <= 0,
            df["Cobertura_Dias"] < df["Cob_Min_Aplicada"],
            (df["Cobertura_Dias"] > dias_excesso) & (df["Cobertura_Dias"] < 999.0),
            df["Cobertura_Dias"] == 999.0,
        ],
        [
            "🔴 Ruptura Atual",
            "🟡 Atenção (Baixo)",
            "🔵 Excesso",
            "⚪ Sem Saída no Período",
        ],
        default="🟢 Saudável",
    )

    df["Historico_Ruptura_Tag"] = np.where(
        df["Dias_Ruptura_Hist"] > 0,
        df["Dias_Ruptura_Hist"].astype(int).astype(str) + "d zerado",
        "100% disponível",
    )

    # 12. Sugestão de Compra baseada na Demanda Real
    necessidade_bruta = (df["DMD_Referencia"] * df["Cob_Alvo_Aplicada"]) - df["Saldo_Calculado"]
    df["Sugestao_Compra"] = np.where(
        necessidade_bruta > 0,
        np.ceil(necessidade_bruta),
        0.0,
    )

    return df, df_kardex


def _gerar_resumo_agrupado(df_filtrado: pd.DataFrame, coluna_grupo: str) -> pd.DataFrame:
    """Agrupa os indicadores históricos, de margem e projeções futuras por Categoria ou Marca."""
    if df_filtrado.empty or coluna_grupo not in df_filtrado.columns:
        return pd.DataFrame()

    registros = []
    for grupo, g in df_filtrado.groupby(coluna_grupo, dropna=False):
        fat_realizado = float(g["Valor_Saidas"].sum())
        perda_hist = float(g["Perda_Financeira_Hist"].sum())
        receita_pot = fat_realizado + perda_hist
        capital_est = float(g["Valor_Estoque_Atual"].sum())

        g_com_custo = g[g["Tem_Custo_XML"]]
        fat_com_custo = float(g_com_custo["Valor_Saidas"].sum())
        lucro_bruto = float(g_com_custo["Lucro_Bruto"].sum())
        lucro_liq = float(g_com_custo["Lucro_Liquido_Est"].sum())
        lucro_perdido = float(g_com_custo["Lucro_Bruto_Perdido_Hist"].sum())

        margem_bruta_pct = round((lucro_bruto / fat_com_custo) * 100, 1) if fat_com_custo > 0 else np.nan
        margem_liq_pct = round((lucro_liq / fat_com_custo) * 100, 1) if fat_com_custo > 0 else np.nan

        registros.append(
            {
                coluna_grupo: grupo,
                "Qtd_SKUs": int(len(g)),
                "SKUs_Ruptura_Atual": int((g["Saldo_Calculado"] <= 0).sum()),
                "SKUs_Faltaram_Passado": int((g["Dias_Ruptura_Hist"] > 0).sum()),
                "Disponibilidade_Media_Pct": round(float(g["Disponibilidade_Pct"].mean()), 1),
                "Faturamento_Realizado": round(fat_realizado, 2),
                "Lucro_Bruto": round(lucro_bruto, 2),
                "Margem_Bruta_Pct": margem_bruta_pct,
                "Lucro_Liquido_Est": round(lucro_liq, 2),
                "Margem_Liquida_Pct": margem_liq_pct,
                "Venda_Perdida_Passado": round(perda_hist, 2),
                "Lucro_Bruto_Perdido": round(lucro_perdido, 2),
                "Receita_Potencial": round(receita_pot, 2),
                "Capital_Estoque": round(capital_est, 2),
                "Giro_Medio": round(float(g["Giro_Estoque"].mean()), 2),
                "Sugestao_Compra_Un": int(g["Sugestao_Compra"].sum()),
            }
        )

    resumo = pd.DataFrame(registros).sort_values(by="Receita_Potencial", ascending=False).reset_index(drop=True)
    return resumo


def _gerar_base_proximo_ciclo(
    df_ind: pd.DataFrame,
    data_registro: str,
    faturamento_total: float,
    fat_medio_diario: float,
    dias_periodo: int,
    valor_estoque_total: float,
    qtd_ruptura: int,
    perda_ruptura_hist: float,
) -> pd.DataFrame:
    """Monta a planilha estruturada para continuar a análise no próximo ciclo preservando o histórico."""
    colunas_preservar = [c for c in ["SKU", "Descrição", "Categoria", "Marca", "Fator_Conversao"] if c in df_ind.columns]
    df_next = df_ind[colunas_preservar].copy()
    df_next["Qtd Inicial"] = df_ind["Saldo_Calculado"].clip(lower=0.0)
    df_next["Saldo_Anterior_Ref"] = df_ind["Saldo_Calculado"]
    df_next["Status_Anterior_Ref"] = df_ind["Status_Estoque"]
    df_next["Saidas_Acum_Ref"] = df_ind["Saidas_Acumuladas"]
    df_next["Dias_Disp_Acum_Ref"] = df_ind["Dias_Disp_Acumulados"]
    df_next["Ref_Data"] = data_registro
    df_next["Ref_Dias_Periodo"] = int(dias_periodo)
    df_next["Ref_Faturamento"] = round(faturamento_total, 2)
    df_next["Ref_Fat_Medio_Diario"] = round(fat_medio_diario, 2)
    df_next["Ref_Capital"] = round(valor_estoque_total, 2)
    df_next["Ref_Ruptura_Qtd"] = int(qtd_ruptura)
    df_next["Ref_Perda_Ruptura"] = round(perda_ruptura_hist, 2)
    return df_next


def renderizar_aba_indicadores():
    st.header("📊 Indicadores Históricos (Kardex) e Projeção de Ruptura")
    st.caption(
        "Reconstrução dia a dia do passado de cada SKU (identificando rupturas ocultas, demanda real e margens) para projetar o risco futuro."
    )

    if "df_xml_consolidado" not in st.session_state or st.session_state["df_xml_consolidado"] is None or st.session_state["df_xml_consolidado"].empty:
        st.info(
            "Nenhum dado consolidado encontrado. Processe os arquivos XML na **Etapa 2** primeiro."
        )
        return

    df_consolidado = st.session_state["df_xml_consolidado"]
    df_movs = st.session_state.get("df_movs_detalhadas", pd.DataFrame())
    snap = st.session_state.get("ponto_partida")
    tem_comparacao = snap is not None

    # 1. Parâmetros da Simulação + Regras por Categoria
    with st.container(border=True):
        st.subheader("Parâmetros da Simulação")
        p1, p2, p3, p4, p5 = st.columns(5)
        dias_periodo = p1.number_input(
            "Janela analisada (dias)",
            min_value=1,
            max_value=365,
            value=30,
            help="Quantidade de dias corridos coberta pelos XMLs importados.",
        )
        cobertura_minima = p2.number_input(
            "Cob. mínima padrão (dias)",
            min_value=1,
            max_value=90,
            value=7,
            help="Padrão global de alerta. Pode ser personalizado por categoria abaixo.",
        )
        cobertura_alvo = p3.number_input(
            "Cob. alvo padrão (dias)",
            min_value=1,
            max_value=180,
            value=15,
            help="Padrão global de compra. Pode ser personalizado por categoria abaixo.",
        )
        dias_excesso = p4.number_input(
            "Limite excesso (dias)",
            min_value=15,
            max_value=365,
            value=45,
            help="Acima deste limite o estoque é considerado excessivo.",
        )
        pct_deducoes_liq = p5.number_input(
            "Impostos + Despesas (%)",
            min_value=0.0,
            max_value=90.0,
            value=15.0,
            step=1.0,
            help="Percentual sobre o faturamento (impostos, taxas de cartão e despesas operacionais) para estimar a Margem Líquida.",
        )

        usar_dmd_acumulada = True
        if tem_comparacao:
            usar_dmd_acumulada = st.checkbox(
                "Usar DMD Real Acumulada (recomendado quando os lotes têm durações diferentes, ex: 8d -> 4d -> 5d)",
                value=True,
            )

        if "Categoria" in df_consolidado.columns:
            cats_serie = df_consolidado["Categoria"].fillna("Geral").astype(str).str.strip().replace("", "Geral")
        else:
            cats_serie = pd.Series(["Geral"])
        categorias_unicas = sorted(cats_serie.unique().tolist())

        regras_categoria = {}
        with st.expander("⚙️ Configurar Cobertura Mínima e Cobertura Alvo por Categoria (opcional)", expanded=False):
            st.caption(
                "Personalize os dias de **Cobertura Mínima (Alerta)** e **Cobertura Alvo (Compra)** para cada categoria conforme o prazo de entrega dos fornecedores."
            )
            df_regras_cat = pd.DataFrame(
                {
                    "Categoria": categorias_unicas,
                    "Cobertura Mínima / Alerta (dias)": [int(cobertura_minima)] * len(categorias_unicas),
                    "Cobertura Alvo Compra (dias)": [int(cobertura_alvo)] * len(categorias_unicas),
                }
            )
            df_regras_editado = st.data_editor(
                df_regras_cat,
                disabled=["Categoria"],
                column_config={
                    "Categoria": st.column_config.TextColumn("Categoria"),
                    "Cobertura Mínima / Alerta (dias)": st.column_config.NumberColumn(
                        "Cobertura Mínima / Alerta (dias)",
                        min_value=1,
                        max_value=180,
                        step=1,
                        format="%d d",
                    ),
                    "Cobertura Alvo Compra (dias)": st.column_config.NumberColumn(
                        "Cobertura Alvo Compra (dias)",
                        min_value=1,
                        max_value=365,
                        step=1,
                        format="%d d",
                    ),
                },
                use_container_width=True,
                hide_index=True,
                key="editor_cobertura_categoria",
            )
            for _, row_cat in df_regras_editado.iterrows():
                val_min = row_cat["Cobertura Mínima / Alerta (dias)"]
                val_alvo = row_cat["Cobertura Alvo Compra (dias)"]
                regras_categoria[str(row_cat["Categoria"])] = {
                    "min": int(val_min) if pd.notna(val_min) else int(cobertura_minima),
                    "alvo": int(val_alvo) if pd.notna(val_alvo) else int(cobertura_alvo),
                }

    df_ind, df_kardex = calcular_indicadores_estoque(
        df_consolidado=df_consolidado,
        df_movs=df_movs,
        dias_periodo=int(dias_periodo),
        cobertura_minima=int(cobertura_minima),
        cobertura_alvo=int(cobertura_alvo),
        dias_excesso=int(dias_excesso),
        pct_deducoes_liq=float(pct_deducoes_liq),
        regras_categoria=regras_categoria,
        snap_anterior=snap,
        usar_dmd_acumulada=usar_dmd_acumulada,
    )

    dias_reais_kardex = int(df_ind["Dias_Janela_Hist"].max()) if not df_ind.empty else max(int(dias_periodo), 1)
    faturamento_total = float(df_ind["Valor_Saidas"].sum())
    fat_medio_diario = faturamento_total / max(dias_reais_kardex, 1)
    valor_estoque_total = float(df_ind["Valor_Estoque_Atual"].sum())
    qtd_ruptura_atual = int((df_ind["Saldo_Calculado"] <= 0).sum())
    qtd_ruptura_passado = int((df_ind["Dias_Ruptura_Hist"] > 0).sum())
    perda_ruptura_hist = float(df_ind["Perda_Financeira_Hist"].sum())
    disponibilidade_media = float(df_ind["Disponibilidade_Pct"].mean()) if not df_ind.empty else 100.0
    total_sugerido = int(df_ind["Sugestao_Compra"].sum())

    # Cálculo das margens consolidadas (considerando itens com custo de compra identificado nos XMLs)
    df_com_custo = df_ind[df_ind["Tem_Custo_XML"]]
    fat_com_custo = float(df_com_custo["Valor_Saidas"].sum())
    lucro_bruto_total = float(df_com_custo["Lucro_Bruto"].sum())
    lucro_liq_total = float(df_com_custo["Lucro_Liquido_Est"].sum())
    margem_bruta_geral = (lucro_bruto_total / fat_com_custo) * 100 if fat_com_custo > 0 else None
    margem_liq_geral = (lucro_liq_total / fat_com_custo) * 100 if fat_com_custo > 0 else None

    # 2. Cards de Resumo Executivo (Histórico + Rentabilidade + Projeção)
    st.subheader("Resumo Executivo (Passado Reconstruído & Projeção)")
    if tem_comparacao:
        dias_p1 = int(snap.get("dias_periodo", dias_reais_kardex))
        st.caption(
            f"📈 Kardex reconstruído em **{dias_reais_kardex} dia(s)** | Comparando contra **Ponto de Partida** de **{snap['data_registro']}** ({dias_p1} dia(s))."
        )
    else:
        st.caption(f"📅 Linha do tempo reconstruída em **{dias_reais_kardex} dia(s)** a partir dos XMLs.")

    delta_fat_dia = None
    delta_cap = None
    delta_rup_qtd = None
    delta_rup_val = None

    if tem_comparacao:
        fat_dia_p1 = float(snap.get("fat_medio_diario", 0.0))
        if fat_dia_p1 <= 0 and float(snap.get("faturamento_total", 0.0)) > 0:
            fat_dia_p1 = float(snap.get("faturamento_total", 0.0)) / max(int(snap.get("dias_periodo", dias_reais_kardex)), 1)

        diff_fat_dia = fat_medio_diario - fat_dia_p1
        diff_cap = valor_estoque_total - float(snap.get("valor_estoque_total", 0.0))
        diff_rup_qtd = qtd_ruptura_atual - int(snap.get("qtd_ruptura", 0))
        diff_rup_val = perda_ruptura_hist - float(snap.get("perda_ruptura_total", 0.0))

        delta_fat_dia = f"{_formatar_brl(diff_fat_dia)}/dia vs. P1"
        delta_cap = f"{_formatar_brl(diff_cap)} vs. P1"
        delta_rup_qtd = f"{diff_rup_qtd:+d} SKU(s) vs. P1"
        delta_rup_val = f"{_formatar_brl(diff_rup_val)} vs. P1"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Faturamento Médio / Dia",
        f"{_formatar_brl(fat_medio_diario)}/dia",
        delta=delta_fat_dia if tem_comparacao else f"Total: {_formatar_brl(faturamento_total)}",
        delta_color="normal" if tem_comparacao else "off",
    )
    m2.metric(
        "Margem Bruta (XML)",
        _formatar_pct(margem_bruta_geral),
        delta=f"Lucro Bruto: {_formatar_brl(lucro_bruto_total)}" if margem_bruta_geral is not None else "Importe XML de compra",
        delta_color="normal" if margem_bruta_geral is not None else "off",
        help="Calculada comparando o preço de venda (NF saída) com o custo médio de compra (NF entrada).",
    )
    m3.metric(
        "Margem Líquida Est.",
        _formatar_pct(margem_liq_geral),
        delta=f"Lucro Líq.: {_formatar_brl(lucro_liq_total)}" if margem_liq_geral is not None else f"Deduções: {pct_deducoes_liq:.1f}%",
        delta_color="normal" if margem_liq_geral is not None else "off",
        help=f"Margem Bruta descontada de {pct_deducoes_liq:.1f}% de impostos e despesas operacionais.",
    )
    m4.metric(
        "Capital em Estoque",
        _formatar_brl(valor_estoque_total),
        delta=delta_cap,
        delta_color="off",
    )

    m5, m6, m7, m8 = st.columns(4)
    m5.metric(
        "Disponibilidade Gôndola",
        f"{disponibilidade_media:.1f}%",
        delta=f"{qtd_ruptura_passado} SKU(s) faltaram no período",
        delta_color="inverse" if qtd_ruptura_passado > 0 else "off",
    )
    m6.metric(
        "Venda Perdida (Passado)",
        _formatar_brl(perda_ruptura_hist),
        delta=delta_rup_val if tem_comparacao else None,
        delta_color="inverse",
        help="Faturamento perdido nos dias em que os produtos ficaram zerados dentro do período analisado.",
    )
    m7.metric(
        "Ruptura Hoje (Fim Período)",
        f"{qtd_ruptura_atual} SKU(s)",
        delta=delta_rup_qtd,
        delta_color="inverse",
    )
    m8.metric("Sugestão Compra (Futura)", f"{total_sugerido} un.")

    # 3. Controle de Ciclos: Salvar Ponto de Partida
    with st.expander("📌 Ponto de Partida (Salvar Snapshot para Comparar Próximos Ciclos)", expanded=False):
        st.write(
            "Salve o estado atual como **Ponto de Partida (Ponto 1)**. Na próxima simulação, "
            "o saldo final de hoje será usado como estoque inicial e a **DMD Real** será acumulada."
        )
        c_snap1, c_snap2 = st.columns(2)
        data_agora = datetime.now().strftime("%d/%m/%Y %H:%M")
        df_next_cycle = _gerar_base_proximo_ciclo(
            df_ind=df_ind,
            data_registro=data_agora,
            faturamento_total=faturamento_total,
            fat_medio_diario=fat_medio_diario,
            dias_periodo=dias_reais_kardex,
            valor_estoque_total=valor_estoque_total,
            qtd_ruptura=qtd_ruptura_atual,
            perda_ruptura_hist=perda_ruptura_hist,
        )

        if c_snap1.button("💾 Salvar Estado Atual como Ponto de Partida", type="secondary", use_container_width=True):
            st.session_state["ponto_partida"] = {
                "data_registro": data_agora,
                "dias_periodo": dias_reais_kardex,
                "faturamento_total": faturamento_total,
                "fat_medio_diario": fat_medio_diario,
                "valor_estoque_total": valor_estoque_total,
                "qtd_ruptura": qtd_ruptura_atual,
                "perda_ruptura_total": perda_ruptura_hist,
                "df_base_proximo_ciclo": df_next_cycle,
                "df_indicadores_ref": df_ind.copy(),
            }
            st.success(
                f"✅ Ponto de Partida salvo ({data_agora})! Volte na **Etapa 2** e selecione "
                "'Continuar do Ponto de Partida salvo' quando quiser processar os próximos XMLs."
            )
            st.rerun()

        csv_proximo_ciclo = df_next_cycle.to_csv(index=False, sep=";").encode("utf-8-sig")
        data_arq = datetime.now().strftime("%Y-%m-%d_%H-%M")
        c_snap2.download_button(
            label="📤 Baixar Base p/ Próximo Ciclo (.CSV)",
            data=csv_proximo_ciclo,
            file_name=f"base_proximo_ciclo_{data_arq}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # 4. Painel Dedicado: Ruptura Histórica (Passado) & Risco Futuro
    df_criticos = df_ind[
        (df_ind["Dias_Ruptura_Hist"] > 0)
        | (df_ind["Status_Estoque"].isin(["🔴 Ruptura Atual", "🟡 Atenção (Baixo)"]))
    ].sort_values(by=["Perda_Financeira_Hist", "Valor_Saidas"], ascending=False)

    if not df_criticos.empty:
        with st.container(border=True):
            st.subheader(f"🚨 Auditoria de Ruptura (Passado Oculto + Risco Futuro) — {len(df_criticos)} item(ns)")
            st.caption(
                "Inclui tanto itens que **sofreram falta no meio do período** (mesmo já repostos hoje) quanto produtos em **risco futuro iminente**."
            )
            cols_ruptura = [
                c
                for c in [
                    "SKU",
                    "Descrição",
                    "Categoria",
                    "Marca",
                    "Curva_ABC",
                    "Status_Estoque",
                    "Previsao_Ruptura",
                    "Dias_Ruptura_Hist",
                    "Disponibilidade_Pct",
                    "DMD_Real_Hist",
                    "Margem_Bruta_Pct",
                    "Perda_Financeira_Hist",
                    "Lucro_Bruto_Perdido_Hist",
                    "Saldo_Calculado",
                    "Sugestao_Compra",
                ]
                if c in df_criticos.columns
            ]
            st.dataframe(
                df_criticos[cols_ruptura].rename(
                    columns={
                        "Curva_ABC": "Curva",
                        "Status_Estoque": "Situação Hoje",
                        "Previsao_Ruptura": "Previsão Ruptura",
                        "Dias_Ruptura_Hist": "Dias Zerado (Passado)",
                        "Disponibilidade_Pct": "Disponib. (%)",
                        "DMD_Real_Hist": "DMD Real",
                        "Margem_Bruta_Pct": "Margem Bruta (%)",
                        "Perda_Financeira_Hist": "Venda Perdida Passado (R\$)",
                        "Lucro_Bruto_Perdido_Hist": "Lucro Bruto Perdido (R\$)",
                        "Saldo_Calculado": "Saldo Hoje",
                        "Sugestao_Compra": "Comprar Agora (un.)",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            csv_compra_urgente = df_criticos[cols_ruptura].to_csv(index=False, sep=";").encode("utf-8-sig")
            st.download_button(
                label="🛒 Baixar Lista de Ruptura Histórica e Reposição (.CSV)",
                data=csv_compra_urgente,
                file_name=f"auditoria_ruptura_reposicao_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # 5. Filtros Rápidos Globais para as Abas de Análise
    st.subheader("Análise Detalhada e Consolidações")
    f1, f2, f3, f4 = st.columns(4)
    filtro_curva = f1.multiselect(
        "Curva ABC (Potencial):",
        options=["A", "B", "C"],
        default=["A", "B", "C"],
    )
    filtro_status = f2.multiselect(
        "Situação Futura / Hoje:",
        options=sorted(df_ind["Status_Estoque"].unique().tolist()),
        default=sorted(df_ind["Status_Estoque"].unique().tolist()),
    )
    opcoes_cat = sorted(df_ind["Categoria"].unique().tolist())
    filtro_categoria = f3.multiselect(
        "Categoria:",
        options=opcoes_cat,
        default=opcoes_cat,
    )
    opcoes_marca = sorted(df_ind["Marca"].unique().tolist())
    filtro_marca = f4.multiselect(
        "Marca:",
        options=opcoes_marca,
        default=opcoes_marca,
    )

    df_filtrado = df_ind[
        df_ind["Curva_ABC"].isin(filtro_curva)
        & df_ind["Status_Estoque"].isin(filtro_status)
        & df_ind["Categoria"].isin(filtro_categoria)
        & df_ind["Marca"].isin(filtro_marca)
    ].copy()

    # Adiciona comparação por SKU quando existir Ponto de Partida
    if tem_comparacao and "df_base_proximo_ciclo" in snap:
        df_ref = snap["df_base_proximo_ciclo"]
        if "SKU" in df_ref.columns and "Saldo_Anterior_Ref" in df_ref.columns:
            mapa_saldo_ant = dict(zip(df_ref["SKU"].astype(str), pd.to_numeric(df_ref["Saldo_Anterior_Ref"], errors="coerce").fillna(0.0)))
            df_filtrado["Var_Saldo"] = df_filtrado["SKU"].astype(str).map(mapa_saldo_ant)
            df_filtrado["Var_Saldo"] = (df_filtrado["Saldo_Calculado"] - df_filtrado["Var_Saldo"]).round(2)

    tab_sku, tab_kardex, tab_agrupado, tab_graficos = st.tabs(
        [
            "📦 Detalhamento por SKU",
            "🔍 Inspetor de Extrato Diário (Kardex)",
            "🏷️ Consolidado por Categoria e Marca",
            "📈 Evolução Diária Geral",
        ]
    )

    # --- SUB-ABA 1: DETALHAMENTO POR SKU ---
    with tab_sku:
        df_exibicao = df_filtrado.copy()
        df_exibicao["Cobertura_Exibicao"] = np.where(
            df_exibicao["Cobertura_Dias"] >= 999.0,
            f"Sem giro (>{dias_reais_kardex}d)",
            df_exibicao["Cobertura_Dias"].astype(str) + " d",
        )

        colunas_ordenadas = [
            "SKU",
            "Descrição",
            "Categoria",
            "Marca",
            "Curva_ABC",
            "Status_Estoque",
            "Previsao_Ruptura",
            "Saldo_Calculado",
            "Var_Saldo",
            "Dias_Ruptura_Hist",
            "Disponibilidade_Pct",
            "DMD_Real_Hist",
            "DMD_Real_Acumulada",
            "DMD_Bruta_Lote",
            "Cobertura_Exibicao",
            "Cob_Min_Aplicada",
            "Cob_Alvo_Aplicada",
            "Total_Saidas",
            "Valor_Saidas",
            "Lucro_Bruto",
            "Margem_Bruta_Pct",
            "Lucro_Liquido_Est",
            "Margem_Liquida_Pct",
            "Perda_Financeira_Hist",
            "Giro_Estoque",
            "Sugestao_Compra",
        ]
        if not tem_comparacao and "DMD_Real_Acumulada" in colunas_ordenadas:
            colunas_ordenadas.remove("DMD_Real_Acumulada")

        colunas_presentes = [c for c in colunas_ordenadas if c in df_exibicao.columns]

        st.dataframe(
            df_exibicao[colunas_presentes].rename(
                columns={
                    "Curva_ABC": "Curva ABC",
                    "Status_Estoque": "Situação Hoje",
                    "Previsao_Ruptura": "Previsão Ruptura",
                    "Saldo_Calculado": "Saldo Hoje",
                    "Var_Saldo": "Var. Saldo (vs P1)",
                    "Dias_Ruptura_Hist": "Dias Zerado (Passado)",
                    "Disponibilidade_Pct": "Disponib. (%)",
                    "DMD_Real_Hist": "DMD Real",
                    "DMD_Real_Acumulada": "DMD Real Acum.",
                    "DMD_Bruta_Lote": "DMD Bruta",
                    "Cobertura_Exibicao": "Cobertura Futura",
                    "Cob_Min_Aplicada": "Cob. Mín. (d)",
                    "Cob_Alvo_Aplicada": "Cob. Alvo (d)",
                    "Total_Saidas": "Saídas (Qtd)",
                    "Valor_Saidas": "Faturamento (R\$)",
                    "Lucro_Bruto": "Lucro Bruto (R\$)",
                    "Margem_Bruta_Pct": "Margem Bruta (%)",
                    "Lucro_Liquido_Est": "Lucro Líq. Est. (R\$)",
                    "Margem_Liquida_Pct": "Margem Líq. (%)",
                    "Perda_Financeira_Hist": "Venda Perdida Passado (R\$)",
                    "Giro_Estoque": "Giro Real",
                    "Sugestao_Compra": "Sugestão Compra",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        data_hoje = datetime.now().strftime("%Y-%m-%d_%H-%M")
        csv_indicadores = df_filtrado.to_csv(index=False, sep=";").encode("utf-8-sig")
        st.download_button(
            label="📥 Baixar Relatório Completo de Indicadores por SKU (.CSV)",
            data=csv_indicadores,
            file_name=f"indicadores_kardex_sku_{data_hoje}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

    # --- SUB-ABA 2: INSPETOR DE EXTRATO DIÁRIO (KARDEX) ---
    with tab_kardex:
        st.markdown("#### Auditoria Dia a Dia por Produto")
        st.caption(
            "Selecione um SKU para visualizar exatamente em quais datas o saldo zerou, quando entraram notas de compra e como foi o ritmo diário de vendas."
        )

        if df_filtrado.empty or df_kardex.empty:
            st.info("Nenhum SKU disponível com os filtros selecionados.")
        else:
            opcoes_sku_label = [
                f"{r['SKU']} — {r['Descrição']} ({r['Categoria']} | {r['Marca']})"
                for _, r in df_filtrado.iterrows()
            ]
            mapa_label_sku = dict(zip(opcoes_sku_label, df_filtrado["SKU"].astype(str)))

            sku_escolhido_label = st.selectbox(
                "Escolha o produto para inspecionar o extrato diário:",
                options=opcoes_sku_label,
                key="select_inspetor_kardex",
            )
            sku_sel = mapa_label_sku[sku_escolhido_label]

            row_sku = df_filtrado[df_filtrado["SKU"].astype(str) == sku_sel].iloc[0]
            df_k_sku = df_kardex[df_kardex["SKU"].astype(str) == sku_sel].copy()

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Dias com Estoque", f"{int(row_sku['Dias_Com_Estoque'])} / {dias_reais_kardex} d")
            k2.metric("Dias Zerado (Passado)", f"{int(row_sku['Dias_Ruptura_Hist'])} d")
            k3.metric("DMD Real (Dias c/ Estoque)", f"{row_sku['DMD_Real_Hist']:.2f} un./dia")
            k4.metric(
                "Margem Bruta / Líq.",
                f"{_formatar_pct(row_sku['Margem_Bruta_Pct'])} / {_formatar_pct(row_sku['Margem_Liquida_Pct'])}",
            )
            k5.metric("Venda Perdida no Passado", _formatar_brl(float(row_sku["Perda_Financeira_Hist"])))

            st.line_chart(
                df_k_sku.set_index("Data")["Saldo_Fechamento"],
                use_container_width=True,
            )

            st.dataframe(
                df_k_sku[
                    [
                        "Data",
                        "Saldo_Abertura",
                        "Entradas",
                        "Saidas",
                        "Saldo_Fechamento",
                        "Situacao_Dia",
                    ]
                ].rename(
                    columns={
                        "Saldo_Abertura": "Saldo Abertura",
                        "Entradas": "Entradas (+)",
                        "Saidas": "Saídas (-)",
                        "Saldo_Fechamento": "Saldo Fechamento",
                        "Situacao_Dia": "Situação no Dia",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            csv_kardex_sku = df_k_sku.to_csv(index=False, sep=";").encode("utf-8-sig")
            st.download_button(
                label=f"📥 Baixar Extrato Diário do SKU {sku_sel} (.CSV)",
                data=csv_kardex_sku,
                file_name=f"extrato_diario_{sku_sel}_{datetime.now().strftime('%Y-%m-%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # --- SUB-ABA 3: CONSOLIDADO POR CATEGORIA E MARCA ---
    with tab_agrupado:
        st.markdown("#### Visão Executiva por Categoria e por Marca")
        tipo_agrupamento = st.radio(
            "Agrupar indicadores por:",
            options=["Categoria", "Marca"],
            horizontal=True,
            key="radio_agrupamento_cat_marca",
        )

        df_grupo = _gerar_resumo_agrupado(df_filtrado, tipo_agrupamento)

        if df_grupo.empty:
            st.info("Sem dados disponíveis para agrupar com os filtros atuais.")
        else:
            top_fat = df_grupo.sort_values("Faturamento_Realizado", ascending=False).iloc[0]
            top_perda = df_grupo.sort_values("Venda_Perdida_Passado", ascending=False).iloc[0]
            df_grupo_margem = df_grupo.dropna(subset=["Margem_Bruta_Pct"])

            g1, g2, g3 = st.columns(3)
            g1.metric(
                f"Líder em Faturamento ({tipo_agrupamento})",
                str(top_fat[tipo_agrupamento]),
                delta=_formatar_brl(float(top_fat["Faturamento_Realizado"])),
                delta_color="off",
            )
            if not df_grupo_margem.empty:
                top_margem = df_grupo_margem.sort_values("Lucro_Bruto", ascending=False).iloc[0]
                g2.metric(
                    f"Maior Lucro Bruto ({tipo_agrupamento})",
                    str(top_margem[tipo_agrupamento]),
                    delta=f"{_formatar_brl(float(top_margem['Lucro_Bruto']))} ({_formatar_pct(top_margem['Margem_Bruta_Pct'])})",
                    delta_color="off",
                )
            else:
                g2.metric(
                    f"Maior Lucro Bruto ({tipo_agrupamento})",
                    "Sem custo XML",
                    delta="Importe XML de compra",
                    delta_color="off",
                )
            g3.metric(
                f"Maior Perda por Ruptura ({tipo_agrupamento})",
                str(top_perda[tipo_agrupamento]),
                delta=f"-{_formatar_brl(float(top_perda['Venda_Perdida_Passado']))} perdidos" if top_perda["Venda_Perdida_Passado"] > 0 else "Sem perdas",
                delta_color="inverse" if top_perda["Venda_Perdida_Passado"] > 0 else "off",
            )

            st.caption(f"Comparativo de Faturamento Realizado vs. Lucro Bruto vs. Venda Perdida por {tipo_agrupamento}")
            chart_grupo = df_grupo.set_index(tipo_agrupamento)[
                ["Faturamento_Realizado", "Lucro_Bruto", "Venda_Perdida_Passado"]
            ].rename(
                columns={
                    "Faturamento_Realizado": "Faturamento Realizado (R$)",
                    "Lucro_Bruto": "Lucro Bruto (R$)",
                    "Venda_Perdida_Passado": "Venda Perdida no Passado (R$)",
                }
            )
            st.bar_chart(chart_grupo, use_container_width=True)

            st.dataframe(
                df_grupo.rename(
                    columns={
                        "Qtd_SKUs": "SKUs",
                        "SKUs_Ruptura_Atual": "Ruptura Hoje",
                        "SKUs_Faltaram_Passado": "Faltaram no Passado",
                        "Disponibilidade_Media_Pct": "Disponib. Média (%)",
                        "Faturamento_Realizado": "Faturamento (R\$)",
                        "Lucro_Bruto": "Lucro Bruto (R\$)",
                        "Margem_Bruta_Pct": "Margem Bruta (%)",
                        "Lucro_Liquido_Est": "Lucro Líq. Est. (R\$)",
                        "Margem_Liquida_Pct": "Margem Líq. (%)",
                        "Venda_Perdida_Passado": "Venda Perdida (R\$)",
                        "Lucro_Bruto_Perdido": "Lucro Bruto Perdido (R\$)",
                        "Receita_Potencial": "Receita Potencial (R\$)",
                        "Capital_Estoque": "Capital Estoque (R\$)",
                        "Giro_Medio": "Giro Médio",
                        "Sugestao_Compra_Un": "Compra Sugerida (un.)",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            csv_grupo = df_grupo.to_csv(index=False, sep=";").encode("utf-8-sig")
            st.download_button(
                label=f"📥 Baixar Consolidado por {tipo_agrupamento} (.CSV)",
                data=csv_grupo,
                file_name=f"consolidado_{tipo_agrupamento.lower()}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # --- SUB-ABA 4: EVOLUÇÃO DIÁRIA GERAL ---
    with tab_graficos:
        col_g1, col_g2 = st.columns(2)
        if df_movs is not None and not df_movs.empty and "Data_Emissao" in df_movs.columns:
            skus_filtrados_set = set(df_filtrado["SKU"].astype(str))
            df_saidas_dia = df_movs[
                (df_movs["Tipo"] == "Saída")
                & (df_movs["Data_Emissao"] != "")
                & (df_movs["SKU"].astype(str).isin(skus_filtrados_set))
            ].copy()
            if not df_saidas_dia.empty:
                with col_g1:
                    st.markdown("#### Volume Diário de Vendas (Unidades)")
                    resumo_diario = (
                        df_saidas_dia.groupby("Data_Emissao", as_index=False)["Quantidade"]
                        .sum()
                        .sort_values("Data_Emissao")
                    )
                    st.bar_chart(resumo_diario.set_index("Data_Emissao")["Quantidade"])

        if not df_kardex.empty and not df_filtrado.empty:
            skus_filtrados_set = set(df_filtrado["SKU"].astype(str))
            df_kardex_filt = df_kardex[df_kardex["SKU"].astype(str).isin(skus_filtrados_set)]
            if not df_kardex_filt.empty:
                with col_g2:
                    st.markdown("#### Histórico Diário de Ruptura (SKUs Zerados/Dia)")
                    ruptura_por_dia = (
                        df_kardex_filt.groupby("Data", as_index=False)["Em_Ruptura"]
                        .sum()
                        .sort_values("Data")
                    )
                    st.line_chart(ruptura_por_dia.set_index("Data")["Em_Ruptura"])
