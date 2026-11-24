from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st


def _formatar_brl(valor: float) -> str:
    """Formata um número float para moeda brasileira (R$ 1.234,56)."""
    sinal = "-" if valor < 0 else ""
    abs_val = abs(valor)
    return f"{sinal}R$ {abs_val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _reconstruir_kardex_diario(
    df: pd.DataFrame,
    df_movs: pd.DataFrame | None,
    col_inicial: str,
    dias_periodo: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconstroi a linha do tempo dia a dia (extrato/kardex diário) de cada SKU
    a partir do saldo inicial e das movimentações datadas dos XMLs.

    Retorna:
      1. DataFrame com métricas históricas por SKU (Dias_Com_Estoque, Dias_Ruptura_Hist,
         Episodios_Ruptura, Estoque_Medio_Diario, etc.)
      2. DataFrame com a série diária consolidada (Data, Saldo_Total, SKUs_Em_Ruptura)
    """
    hoje = pd.Timestamp.now().normalize()
    data_inicio = hoje - pd.Timedelta(days=max(dias_periodo - 1, 0))
    data_fim = hoje

    movs_validas = pd.DataFrame()
    if df_movs is not None and not df_movs.empty and "Data_Emissao" in df_movs.columns:
        temp = df_movs.copy()
        temp["Data_dt"] = pd.to_datetime(temp["Data_Emissao"], errors="coerce").dt.normalize()
        temp = temp.dropna(subset=["Data_dt"])
        if not temp.empty:
            movs_validas = temp
            data_min_xml = temp["Data_dt"].min()
            data_max_xml = temp["Data_dt"].max()
            # Define a janela real ancorada na data máxima dos XMLs e na duração informada
            data_fim = data_max_xml
            data_inicio_calc = data_fim - pd.Timedelta(days=max(dias_periodo - 1, 0))
            data_inicio = min(data_min_xml, data_inicio_calc) if (data_max_xml - data_min_xml).days + 1 == dias_periodo else data_inicio_calc

    calendario = pd.date_range(start=data_inicio, end=data_fim, freq="D")
    total_dias_cal = max(len(calendario), 1)

    # Pré-agrupa entradas e saídas por (SKU, Data_dt)
    entradas_map = {}
    saidas_map = {}
    if not movs_validas.empty:
        movs_validas["SKU_str"] = movs_validas["SKU"].astype(str).str.strip()
        ent_grp = (
            movs_validas[movs_validas["Tipo"] == "Entrada"]
            .groupby(["SKU_str", "Data_dt"])["Quantidade"]
            .sum()
        )
        sai_grp = (
            movs_validas[movs_validas["Tipo"] == "Saída"]
            .groupby(["SKU_str", "Data_dt"])["Quantidade"]
            .sum()
        )
        entradas_map = ent_grp.to_dict()
        saidas_map = sai_grp.to_dict()

    registros_sku = []
    rupturas_por_dia = np.zeros(len(calendario), dtype=int)
    saldo_total_por_dia = np.zeros(len(calendario), dtype=float)

    for _, row in df.iterrows():
        sku = str(row["SKU"]).strip()
        saldo_corrente = float(row.get(col_inicial, 0.0) or 0.0)

        dias_com_estoque = 0
        dias_em_ruptura = 0
        episodios_ruptura = 0
        soma_saldos_diarios = 0.0
        estava_zerado_ontem = saldo_corrente <= 0

        for idx_d, dia in enumerate(calendario):
            ent_dia = float(entradas_map.get((sku, dia), 0.0))
            sai_dia = float(saidas_map.get((sku, dia), 0.0))

            # Disponibilidade do dia considera o saldo de abertura + reposições recebidas no dia
            disponivel_no_dia = saldo_corrente + ent_dia
            saldo_fechamento = disponivel_no_dia - sai_dia

            soma_saldos_diarios += max(saldo_fechamento, 0.0)
            saldo_total_por_dia[idx_d] += max(saldo_fechamento, 0.0)

            # Considera dia com estoque se houve disponibilidade positiva antes/durante as vendas
            if disponivel_no_dia > 0 and saldo_fechamento > 0:
                dias_com_estoque += 1
                estava_zerado_ontem = False
            elif disponivel_no_dia > 0 and saldo_fechamento <= 0:
                # Vendeu até zerar neste dia: o dia teve venda, mas encerrou em ruptura
                dias_com_estoque += 1
                if not estava_zerado_ontem:
                    episodios_ruptura += 1
                estava_zerado_ontem = True
                rupturas_por_dia[idx_d] += 1
            else:
                # Abriu zerado e não recebeu reposição no dia
                dias_em_ruptura += 1
                if not estava_zerado_ontem:
                    episodios_ruptura += 1
                estava_zerado_ontem = True
                rupturas_por_dia[idx_d] += 1

            saldo_corrente = saldo_fechamento

        registros_sku.append(
            {
                "SKU": sku,
                "Dias_Janela_Real": total_dias_cal,
                "Dias_Com_Estoque": dias_com_estoque,
                "Dias_Ruptura_Hist": dias_em_ruptura,
                "Episodios_Ruptura": episodios_ruptura,
                "Nivel_Servico_Disp_Pct": round((dias_com_estoque / total_dias_cal) * 100, 1),
                "Estoque_Medio_Diario": round(soma_saldos_diarios / total_dias_cal, 2),
                "Data_Ref_Fim": data_fim,
            }
        )

    df_kardex_sku = pd.DataFrame(registros_sku)
    df_serie_diaria = pd.DataFrame(
        {
            "Data": calendario.strftime("%Y-%m-%d"),
            "SKUs_Em_Ruptura": rupturas_por_dia,
            "Saldo_Total_Unidades": np.round(saldo_total_por_dia, 2),
        }
    )
    return df_kardex_sku, df_serie_diaria


def calcular_indicadores_estoque(
    df_consolidado: pd.DataFrame,
    df_movs: pd.DataFrame | None = None,
    dias_periodo: int = 30,
    cobertura_minima: int = 7,
    cobertura_alvo: int = 15,
    dias_excesso: int = 45,
    regras_categoria: dict[str, dict[str, int]] | None = None,
    snap_anterior: dict | None = None,
    usar_dmd_acumulada: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula os indicadores ancorados na reconstrução histórica dia a dia (kardex):
      - Passado: Dias com Estoque, Dias Zerado no Período, Episódios de Falta,
        DMD Real (descontando dias sem estoque), Venda Perdida Histórica e Giro Médio Diário.
      - Futuro: Cobertura Futura Real (dias), Data Estimada de Ruptura e Sugestão de Compra.
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

    # 1. Reconstrução Histórica Dia a Dia (Kardex)
    df_kardex, df_serie_diaria = _reconstruir_kardex_diario(
        df=df,
        df_movs=df_movs,
        col_inicial=col_inicial,
        dias_periodo=dias_periodo,
    )
    df = df.merge(df_kardex, on="SKU", how="left")

    data_ref_fim = pd.Timestamp.now().normalize()
    if not df["Data_Ref_Fim"].isna().all():
        data_ref_fim = pd.to_datetime(df["Data_Ref_Fim"].iloc[0])

    # 2. Demanda Média Diária (DMD):
    #    - DMD_Bruta_Lote: Total_Saidas / dias_periodo (diluída pelos dias zerados)
    #    - DMD_Real_Hist: Total_Saidas / Dias_Com_Estoque (velocidade real quando havia produto)
    dias_div_bruto = max(int(dias_periodo), 1)
    df["DMD_Bruta_Lote"] = (df["Total_Saidas"] / dias_div_bruto).round(2)

    df["DMD_Real_Hist"] = np.where(
        df["Dias_Com_Estoque"] > 0,
        (df["Total_Saidas"] / df["Dias_Com_Estoque"]).round(2),
        0.0,
    )

    # 3. Histórico Acumulado entre Ciclos (quando houver Ponto de Partida salvo)
    df["Saidas_Acumuladas"] = df["Total_Saidas"].astype(float)
    df["Dias_Com_Estoque_Acum"] = df["Dias_Com_Estoque"].astype(float)

    if snap_anterior is not None and "df_base_proximo_ciclo" in snap_anterior:
        df_ref = snap_anterior["df_base_proximo_ciclo"]
        if "SKU" in df_ref.columns and "Saidas_Acum_Ref" in df_ref.columns and "Dias_Disp_Acum_Ref" in df_ref.columns:
            mapa_saidas_ant = dict(
                zip(
                    df_ref["SKU"].astype(str).str.strip(),
                    pd.to_numeric(df_ref["Saidas_Acum_Ref"], errors="coerce").fillna(0.0),
                )
            )
            mapa_dias_disp_ant = dict(
                zip(
                    df_ref["SKU"].astype(str).str.strip(),
                    pd.to_numeric(df_ref["Dias_Disp_Acum_Ref"], errors="coerce").fillna(0.0),
                )
            )
            df["Saidas_Acumuladas"] = df["SKU"].map(mapa_saidas_ant).fillna(0.0) + df["Total_Saidas"]
            df["Dias_Com_Estoque_Acum"] = df["SKU"].map(mapa_dias_disp_ant).fillna(0.0) + df["Dias_Com_Estoque"]

    df["DMD_Real_Acumulada"] = np.where(
        df["Dias_Com_Estoque_Acum"] > 0,
        (df["Saidas_Acumuladas"] / df["Dias_Com_Estoque_Acum"]).round(2),
        df["DMD_Real_Hist"],
    )

    # Base ativa de demanda para projeções futuras (Cobertura e Compra)
    df["DMD_Ativa"] = np.where(
        usar_dmd_acumulada,
        df["DMD_Real_Acumulada"],
        df["DMD_Real_Hist"],
    )

    # 4. Estoque Médio Diário Real e Giro de Estoque
    df["Estoque_Medio"] = df["Estoque_Medio_Diario"]
    df["Giro_Estoque"] = np.where(
        df["Estoque_Medio"] > 0,
        (df["Total_Saidas"] / df["Estoque_Medio"]).round(2),
        0.0,
    )

    # 5. Preços Unitários e Valor Estimado em Estoque
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

    # 6. Perda Histórica Real (Passado: dias em que o produto ficou zerado no período)
    df["Unidades_Perdidas_Hist"] = np.where(
        (df["Dias_Ruptura_Hist"] > 0) & (df["DMD_Real_Hist"] > 0),
        np.ceil(df["DMD_Real_Hist"] * df["Dias_Ruptura_Hist"]),
        0.0,
    )
    df["Perda_Financeira_Hist"] = (
        df["Unidades_Perdidas_Hist"] * df["Preco_Medio_Venda"]
    ).round(2)

    # Receita Potencial (Faturado Real + O que deixou de vender nos dias zerados)
    df["Receita_Potencial_Hist"] = (df["Valor_Saidas"] + df["Perda_Financeira_Hist"]).round(2)

    # 7. Curva ABC (considera Receita Potencial para não rebaixar itens que faltaram na prateleira)
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

    # 8. Projeção Futura: Cobertura Real e Previsão de Ruptura
    df["Cobertura_Futura_Dias"] = np.where(
        df["DMD_Ativa"] > 0,
        (df["Saldo_Calculado"].clip(lower=0) / df["DMD_Ativa"]).round(1),
        np.where(df["Saldo_Calculado"] > 0, 999.0, 0.0),
    )

    def _calcular_data_ruptura(row_item) -> str:
        saldo = float(row_item["Saldo_Calculado"])
        cob = float(row_item["Cobertura_Futura_Dias"])
        dmd = float(row_item["DMD_Ativa"])
        if saldo <= 0:
            return "Já em Ruptura"
        if dmd <= 0 or cob >= 999.0:
            return "Sem previsão"
        dt_prev = data_ref_fim + pd.Timedelta(days=int(np.floor(cob)))
        return dt_prev.strftime("%d/%m/%Y")

    df["Previsao_Ruptura_Data"] = df.apply(_calcular_data_ruptura, axis=1)

    df["Risco_Diario_Futuro"] = np.where(
        (df["Saldo_Calculado"] <= 0) & (df["DMD_Ativa"] > 0),
        (df["DMD_Ativa"] * df["Preco_Medio_Venda"]).round(2),
        0.0,
    )

    # 9. Regras de Cobertura Mínima e Cobertura Alvo por Categoria
    regras_norm = {}
    if regras_categoria:
        for cat_nome, cfg in regras_categoria.items():
            regras_norm[str(cat_nome).strip()] = {
                "min": int(cfg.get("min", cobertura_minima)),
                "alvo": int(cfg.get("alvo", cobertura_alvo)),
            }

    df["Cob_Min_Aplicada"] = df["Categoria"].map(
        lambda c: regras_norm.get(c, {}).get("min", int(cobertura_minima))
    )
    df["Cob_Alvo_Aplicada"] = df["Categoria"].map(
        lambda c: regras_norm.get(c, {}).get("alvo", int(cobertura_alvo))
    )

    # 10. Status Operacional Futuro (baseado na Cobertura Futura Real)
    df["Status_Estoque"] = np.select(
        [
            df["Saldo_Calculado"] <= 0,
            df["Cobertura_Futura_Dias"] < df["Cob_Min_Aplicada"],
            (df["Cobertura_Futura_Dias"] > dias_excesso) & (df["Cobertura_Futura_Dias"] < 999.0),
            df["Cobertura_Futura_Dias"] == 999.0,
        ],
        [
            "🔴 Ruptura Atual",
            "🟡 Risco de Ruptura",
            "🔵 Excesso",
            "⚪ Sem Saída no Período",
        ],
        default="🟢 Saudável",
    )

    # Flag de Ruptura Oculta no Passado (item está com estoque hoje, mas ficou zerado durante o período)
    df["Ruptura_Oculta_Passado"] = np.where(
        (df["Saldo_Calculado"] > 0) & (df["Dias_Ruptura_Hist"] > 0),
        "Sim (" + df["Dias_Ruptura_Hist"].astype(int).astype(str) + "d zerado)",
        "Não",
    )

    # 11. Sugestão de Compra Futura (dimensionada pela DMD Real limpa)
    necessidade_bruta = (df["DMD_Ativa"] * df["Cob_Alvo_Aplicada"]) - df["Saldo_Calculado"]
    df["Sugestao_Compra"] = np.where(
        necessidade_bruta > 0,
        np.ceil(necessidade_bruta),
        0.0,
    )

    return df, df_serie_diaria


def _gerar_base_proximo_ciclo(
    df_ind: pd.DataFrame,
    data_registro: str,
    dias_periodo: int,
    faturamento_total: float,
    fat_medio_diario: float,
    valor_estoque_total: float,
    qtd_ruptura: int,
    perda_hist_total: float,
) -> pd.DataFrame:
    """Monta a planilha estruturada para continuar a análise no próximo ciclo."""
    colunas_preservar = [
        c
        for c in ["SKU", "Descrição", "Categoria", "Marca", "Fator_Conversao"]
        if c in df_ind.columns
    ]
    df_next = df_ind[colunas_preservar].copy()
    df_next["Qtd Inicial"] = df_ind["Saldo_Calculado"].clip(lower=0.0)
    df_next["Saldo_Anterior_Ref"] = df_ind["Saldo_Calculado"]
    df_next["Status_Anterior_Ref"] = df_ind["Status_Estoque"]
    df_next["Saidas_Acum_Ref"] = df_ind["Saidas_Acumuladas"]
    df_next["Dias_Disp_Acum_Ref"] = df_ind["Dias_Com_Estoque_Acum"]
    df_next["Ref_Data"] = data_registro
    df_next["Ref_Dias_Periodo"] = int(dias_periodo)
    df_next["Ref_Faturamento"] = round(faturamento_total, 2)
    df_next["Ref_Fat_Medio_Diario"] = round(fat_medio_diario, 2)
    df_next["Ref_Capital"] = round(valor_estoque_total, 2)
    df_next["Ref_Ruptura_Qtd"] = int(qtd_ruptura)
    df_next["Ref_Perda_Ruptura"] = round(perda_hist_total, 2)
    return df_next


def renderizar_aba_indicadores():
    st.header("📊 Indicadores Históricos e Projeção de Ruptura")
    st.caption(
        "Reconstrução dia a dia (kardex) a partir das datas dos XMLs: mede a demanda real sem o viés de dias zerados, "
        "identifica rupturas passadas (inclusive já reabastecidas) e projeta o risco futuro."
    )

    if (
        "df_xml_consolidado" not in st.session_state
        or st.session_state["df_xml_consolidado"] is None
        or st.session_state["df_xml_consolidado"].empty
    ):
        st.info(
            "Nenhum dado consolidado encontrado. Processe os arquivos XML na **Etapa 2** primeiro."
        )
        return

    df_consolidado = st.session_state["df_xml_consolidado"]
    df_movs = st.session_state.get("df_movs_detalhadas", pd.DataFrame())
    snap = st.session_state.get("ponto_partida")
    tem_comparacao = snap is not None

    # 1. Parâmetros de Simulação + Regras por Categoria
    with st.container(border=True):
        st.subheader("Parâmetros da Análise")
        p1, p2, p3, p4 = st.columns(4)
        dias_periodo = p1.number_input(
            "Janela do lote (dias)",
            min_value=1,
            max_value=365,
            value=30,
            help="Quantidade de dias coberta pelos XMLs deste lote (ex: 8 dias, 15 dias, 30 dias).",
        )
        cobertura_minima = p2.number_input(
            "Cob. mínima padrão (dias)",
            min_value=1,
            max_value=90,
            value=7,
            help="Lead time + margem de segurança padrão para alerta de ruptura futura.",
        )
        cobertura_alvo = p3.number_input(
            "Cob. alvo padrão (dias)",
            min_value=1,
            max_value=180,
            value=15,
            help="Dias de estoque desejados na sugestão de compra.",
        )
        dias_excesso = p4.number_input(
            "Limite para excesso (dias)",
            min_value=15,
            max_value=365,
            value=45,
            help="Acima deste limite o estoque é considerado excessivo.",
        )

        usar_dmd_acumulada = False
        if tem_comparacao:
            dias_ant = int(snap.get("dias_periodo", 0) or 0)
            base_dmd_escolha = st.radio(
                "Base de Demanda (DMD Real) para projeção futura e compra:",
                options=[
                    f"DMD Real deste lote ({int(dias_periodo)} dias)",
                    f"DMD Real acumulada ({dias_ant + int(dias_periodo)} dias totais com Ponto de Partida)",
                ],
                horizontal=True,
                help="Use a DMD Real acumulada quando estiver importando lotes curtos ou irregulares (ex: 8d -> 4d -> 5d).",
            )
            usar_dmd_acumulada = base_dmd_escolha.startswith("DMD Real acumulada")

        if "Categoria" in df_consolidado.columns:
            categorias_unicas = sorted(
                df_consolidado["Categoria"]
                .fillna("Geral")
                .astype(str)
                .str.strip()
                .replace("", "Geral")
                .unique()
                .tolist()
            )
        else:
            categorias_unicas = ["Geral"]

        regras_categoria = {}
        with st.expander(
            "⚙️ Configurar Cobertura Mínima e Cobertura Alvo por Categoria",
            expanded=False,
        ):
            st.caption(
                "Personalize os dias de alerta (Cobertura Mínima) e de reposição (Cobertura Alvo) por categoria."
            )
            df_regras_base = pd.DataFrame(
                {
                    "Categoria": categorias_unicas,
                    "Cobertura Mínima / Alerta (dias)": [int(cobertura_minima)] * len(categorias_unicas),
                    "Cobertura Alvo Compra (dias)": [int(cobertura_alvo)] * len(categorias_unicas),
                }
            )
            df_regras_editado = st.data_editor(
                df_regras_base,
                disabled=["Categoria"],
                column_config={
                    "Categoria": st.column_config.TextColumn("Categoria"),
                    "Cobertura Mínima / Alerta (dias)": st.column_config.NumberColumn(
                        "Cob. Mínima (dias)",
                        min_value=1,
                        max_value=180,
                        step=1,
                    ),
                    "Cobertura Alvo Compra (dias)": st.column_config.NumberColumn(
                        "Cob. Alvo Compra (dias)",
                        min_value=1,
                        max_value=365,
                        step=1,
                    ),
                },
                use_container_width=True,
                hide_index=True,
                key="editor_cobertura_categoria_hist",
            )
            for _, row_reg in df_regras_editado.iterrows():
                cat_k = str(row_reg["Categoria"]).strip()
                val_min = int(row_reg["Cobertura Mínima / Alerta (dias)"])
                val_alvo = max(int(row_reg["Cobertura Alvo Compra (dias)"]), val_min)
                regras_categoria[cat_k] = {"min": val_min, "alvo": val_alvo}

    df_ind, df_serie_diaria = calcular_indicadores_estoque(
        df_consolidado=df_consolidado,
        df_movs=df_movs,
        dias_periodo=int(dias_periodo),
        cobertura_minima=int(cobertura_minima),
        cobertura_alvo=int(cobertura_alvo),
        dias_excesso=int(dias_excesso),
        regras_categoria=regras_categoria,
        snap_anterior=snap,
        usar_dmd_acumulada=usar_dmd_acumulada,
    )

    faturamento_total = float(df_ind["Valor_Saidas"].sum())
    fat_medio_diario = faturamento_total / max(int(dias_periodo), 1)
    valor_estoque_total = float(df_ind["Valor_Estoque_Atual"].sum())
    qtd_ruptura_atual = int((df_ind["Saldo_Calculado"] <= 0).sum())
    qtd_ruptura_passada = int((df_ind["Dias_Ruptura_Hist"] > 0).sum())
    perda_hist_total = float(df_ind["Perda_Financeira_Hist"].sum())
    disponibilidade_media = float(df_ind["Nivel_Servico_Disp_Pct"].mean()) if len(df_ind) > 0 else 100.0
    total_sugerido = int(df_ind["Sugestao_Compra"].sum())

    # 2. Cards de Resumo Executivo (Passado + Presente + Futuro)
    st.subheader("Resumo Executivo (Diagnóstico Histórico e Risco Atual)")
    if tem_comparacao:
        dias_p1 = int(snap.get("dias_periodo", 0) or 0)
        info_p1 = f" ({dias_p1}d)" if dias_p1 > 0 else ""
        st.caption(
            f"📈 Comparando lote atual ({int(dias_periodo)}d) contra o **Ponto de Partida** salvo em **{snap['data_registro']}**{info_p1}."
        )

    m1, m2, m3, m4, m5, m6 = st.columns(6)

    delta_fat_dia = None
    delta_cap = None
    delta_rup_qtd = None
    delta_rup_val = None

    if tem_comparacao:
        fat_dia_ant = float(snap.get("fat_medio_diario", 0.0))
        if fat_dia_ant <= 0 and int(snap.get("dias_periodo", 0) or 0) > 0:
            fat_dia_ant = float(snap.get("faturamento_total", 0.0)) / int(snap["dias_periodo"])

        diff_fat_dia = fat_medio_diario - fat_dia_ant
        diff_cap = valor_estoque_total - float(snap.get("valor_estoque_total", 0.0))
        diff_rup_qtd = qtd_ruptura_atual - int(snap.get("qtd_ruptura", 0))
        diff_rup_val = perda_hist_total - float(snap.get("perda_ruptura_total", 0.0))

        delta_fat_dia = f"{_formatar_brl(diff_fat_dia)}/d vs. P1"
        delta_cap = f"{_formatar_brl(diff_cap)} vs. P1"
        delta_rup_qtd = f"{diff_rup_qtd:+d} SKU(s) vs. P1"
        delta_rup_val = f"{_formatar_brl(diff_rup_val)} vs. P1"

    m1.metric(
        "Faturamento Médio/Dia",
        f"{_formatar_brl(fat_medio_diario)}/d",
        delta=delta_fat_dia if tem_comparacao else f"Total: {_formatar_brl(faturamento_total)}",
        delta_color="normal" if tem_comparacao else "off",
    )
    m2.metric(
        "Disponibilidade Histórica",
        f"{disponibilidade_media:.1f}%",
        delta=f"{qtd_ruptura_passada} SKU(s) faltaram no período",
        delta_color="off",
        help="Percentual médio dos dias em que os produtos estiveram disponíveis na prateleira.",
    )
    m3.metric(
        "Venda Perdida (Passado)",
        _formatar_brl(perda_hist_total),
        delta=delta_rup_val,
        delta_color="inverse",
        help="Faturamento estimado deixado na mesa durante os dias em que os SKUs ficaram zerados no período.",
    )
    m4.metric(
        "Ruptura Hoje (Fim Lote)",
        f"{qtd_ruptura_atual} SKU(s)",
        delta=delta_rup_qtd,
        delta_color="inverse",
    )
    m5.metric(
        "Capital em Estoque",
        _formatar_brl(valor_estoque_total),
        delta=delta_cap,
        delta_color="off",
    )
    m6.metric("Sugestão Compra Futura", f"{total_sugerido} un.")

    # 3. Controle de Ciclos: Salvar Ponto de Partida
    with st.expander("📌 Ponto de Partida (Salvar Snapshot para Comparar Próximos Ciclos)", expanded=False):
        st.write(
            "Salve o estado atual como **Ponto de Partida (Ponto 1)**. Na próxima simulação, "
            "o saldo final de hoje será usado como estoque inicial e a **DMD Real** será acumulada sem o viés de dias zerados."
        )
        c_snap1, c_snap2 = st.columns(2)
        data_agora = datetime.now().strftime("%d/%m/%Y %H:%M")
        df_next_cycle = _gerar_base_proximo_ciclo(
            df_ind=df_ind,
            data_registro=data_agora,
            dias_periodo=int(dias_periodo),
            faturamento_total=faturamento_total,
            fat_medio_diario=fat_medio_diario,
            valor_estoque_total=valor_estoque_total,
            qtd_ruptura=qtd_ruptura_atual,
            perda_hist_total=perda_hist_total,
        )

        if c_snap1.button("💾 Salvar Estado Atual como Ponto de Partida", type="secondary", use_container_width=True):
            st.session_state["ponto_partida"] = {
                "data_registro": data_agora,
                "dias_periodo": int(dias_periodo),
                "faturamento_total": faturamento_total,
                "fat_medio_diario": fat_medio_diario,
                "valor_estoque_total": valor_estoque_total,
                "qtd_ruptura": qtd_ruptura_atual,
                "perda_ruptura_total": perda_hist_total,
                "df_base_proximo_ciclo": df_next_cycle,
                "df_indicadores_ref": df_ind.copy(),
            }
            st.success(
                f"✅ Ponto de Partida salvo ({data_agora} — {int(dias_periodo)} dias)! "
                "Volte na **Etapa 2** e selecione 'Continuar do Ponto de Partida salvo' para o próximo lote."
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

    # 4. Painel de Ruptura Histórica (Passado) e Risco Futuro
    df_criticos = df_ind[
        (df_ind["Dias_Ruptura_Hist"] > 0)
        | (df_ind["Status_Estoque"].isin(["🔴 Ruptura Atual", "🟡 Risco de Ruptura"]))
    ].sort_values(
        by=["Perda_Financeira_Hist", "Risco_Diario_Futuro", "Receita_Potencial_Hist"],
        ascending=False,
    )

    if not df_criticos.empty:
        with st.container(border=True):
            st.subheader(f"🚨 Diagnóstico de Ruptura (Passado) e Previsão de Falta ({len(df_criticos)} item(ns))")
            st.caption(
                "Inclui itens que sofreram ruptura no passado (mesmo já reabastecidos), itens zerados hoje e produtos abaixo da cobertura mínima."
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
                    "Dias_Com_Estoque",
                    "Dias_Ruptura_Hist",
                    "Episodios_Ruptura",
                    "DMD_Real_Hist",
                    "Perda_Financeira_Hist",
                    "Saldo_Calculado",
                    "Previsao_Ruptura_Data",
                    "Sugestao_Compra",
                ]
                if c in df_criticos.columns
            ]
            st.dataframe(
                df_criticos[cols_ruptura].rename(
                    columns={
                        "Curva_ABC": "Curva",
                        "Status_Estoque": "Situação Futura",
                        "Dias_Com_Estoque": "Dias c/ Estoque",
                        "Dias_Ruptura_Hist": "Dias Zerado (Passado)",
                        "Episodios_Ruptura": "Vezes Zerou",
                        "DMD_Real_Hist": "DMD Real",
                        "Perda_Financeira_Hist": "Venda Perdida Passado (R$)",
                        "Saldo_Calculado": "Saldo Hoje",
                        "Previsao_Ruptura_Data": "Previsão Ruptura",
                        "Sugestao_Compra": "Comprar Agora (un.)",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            csv_compra_urgente = df_criticos[cols_ruptura].to_csv(index=False, sep=";").encode("utf-8-sig")
            st.download_button(
                label="🛒 Baixar Plano de Ação e Compra (.CSV)",
                data=csv_compra_urgente,
                file_name=f"plano_ruptura_historico_futuro_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # 5. Filtros Rápidos e Tabela Completa (Passado + Presente + Futuro)
    st.subheader("Detalhamento Completo por SKU (Histórico e Projeção)")
    f1, f2, f3, f4 = st.columns(4)
    filtro_curva = f1.multiselect(
        "Curva ABC (Potencial):",
        options=["A", "B", "C"],
        default=["A", "B", "C"],
    )
    filtro_status = f2.multiselect(
        "Situação Futura:",
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

    if tem_comparacao and "df_base_proximo_ciclo" in snap:
        df_ref = snap["df_base_proximo_ciclo"]
        if "SKU" in df_ref.columns and "Saldo_Anterior_Ref" in df_ref.columns:
            mapa_saldo_ant = dict(
                zip(
                    df_ref["SKU"].astype(str).str.strip(),
                    pd.to_numeric(df_ref["Saldo_Anterior_Ref"], errors="coerce").fillna(0.0),
                )
            )
            df_filtrado["Var_Saldo"] = (
                df_filtrado["Saldo_Calculado"] - df_filtrado["SKU"].map(mapa_saldo_ant)
            ).round(2)

    df_exibicao = df_filtrado.copy()
    df_exibicao["Cobertura_Exibicao"] = np.where(
        df_exibicao["Cobertura_Futura_Dias"] >= 999.0,
        f"Sem giro (>{int(dias_periodo)}d)",
        df_exibicao["Cobertura_Futura_Dias"].astype(str) + " d",
    )

    colunas_ordenadas = [
        "SKU",
        "Descrição",
        "Categoria",
        "Marca",
        "Curva_ABC",
        "Status_Estoque",
        "Dias_Com_Estoque",
        "Dias_Ruptura_Hist",
        "Nivel_Servico_Disp_Pct",
        "DMD_Real_Hist",
        "DMD_Real_Acumulada",
        "DMD_Bruta_Lote",
        "Valor_Saidas",
        "Perda_Financeira_Hist",
        "Saldo_Calculado",
        "Var_Saldo",
        "Cobertura_Exibicao",
        "Previsao_Ruptura_Data",
        "Giro_Estoque",
        "Cob_Min_Aplicada",
        "Cob_Alvo_Aplicada",
        "Sugestao_Compra",
    ]
    if not tem_comparacao and "DMD_Real_Acumulada" in colunas_ordenadas:
        colunas_ordenadas.remove("DMD_Real_Acumulada")

    colunas_presentes = [c for c in colunas_ordenadas if c in df_exibicao.columns]

    st.dataframe(
        df_exibicao[colunas_presentes].rename(
            columns={
                "Curva_ABC": "Curva ABC",
                "Status_Estoque": "Situação Futura",
                "Dias_Com_Estoque": "Dias c/ Estoque",
                "Dias_Ruptura_Hist": "Dias Zerado (Hist.)",
                "Nivel_Servico_Disp_Pct": "Disponib. (%)",
                "DMD_Real_Hist": "DMD Real",
                "DMD_Real_Acumulada": "DMD Real Acum.",
                "DMD_Bruta_Lote": "DMD Bruta",
                "Valor_Saidas": "Faturado (R$)",
                "Perda_Financeira_Hist": "Perda Hist. (R$)",
                "Saldo_Calculado": "Saldo Hoje",
                "Var_Saldo": "Var. Saldo (vs P1)",
                "Cobertura_Exibicao": "Cobertura Futura",
                "Previsao_Ruptura_Data": "Previsão Ruptura",
                "Giro_Estoque": "Giro (Méd. Diária)",
                "Cob_Min_Aplicada": "Cob. Mín. (d)",
                "Cob_Alvo_Aplicada": "Cob. Alvo (d)",
                "Sugestao_Compra": "Sugestão Compra",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # 6. Gráficos Históricos: Evolução de Vendas e SKUs Zerados por Dia
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        if df_movs is not None and not df_movs.empty and "Data_Emissao" in df_movs.columns:
            df_saidas_dia = df_movs[
                (df_movs["Tipo"] == "Saída") & (df_movs["Data_Emissao"] != "")
            ].copy()
            if not df_saidas_dia.empty:
                st.subheader("Vendas Diárias (Unidades)")
                resumo_diario = (
                    df_saidas_dia.groupby("Data_Emissao", as_index=False)["Quantidade"]
                    .sum()
                    .sort_values("Data_Emissao")
                )
                st.bar_chart(resumo_diario.set_index("Data_Emissao")["Quantidade"])

    with col_g2:
        if not df_serie_diaria.empty:
            st.subheader("Histórico Diário de SKUs em Ruptura")
            st.line_chart(df_serie_diaria.set_index("Data")["SKUs_Em_Ruptura"])

    # 7. Exportação do Relatório Completo
    data_hoje = datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_indicadores = df_filtrado.to_csv(index=False, sep=";").encode("utf-8-sig")
    st.download_button(
        label="📥 Baixar Relatório Completo (Histórico + Projeção .CSV)",
        data=csv_indicadores,
        file_name=f"indicadores_kardex_estoque_{data_hoje}.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )
