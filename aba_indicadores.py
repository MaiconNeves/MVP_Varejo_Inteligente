from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st


def _formatar_brl(valor: float) -> str:
    """Formata um número float para moeda brasileira (R$ 1.234,56)."""
    sinal = "-" if valor < 0 else ""
    abs_val = abs(valor)
    return f"{sinal}R$ {abs_val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


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
) -> pd.DataFrame:
    """Calcula os principais KPIs de estoque e impacto financeiro da ruptura."""
    df = df_consolidado.copy()

    if "Categoria" not in df.columns:
        df["Categoria"] = "Geral"
    else:
        df["Categoria"] = (
            df["Categoria"]
            .fillna("Geral")
            .astype(str)
            .str.strip()
            .replace({"": "Geral", "nan": "Geral", "None": "Geral"})
        )

    if "Marca" not in df.columns:
        df["Marca"] = "Sem Marca"
    else:
        df["Marca"] = (
            df["Marca"]
            .fillna("Sem Marca")
            .astype(str)
            .str.strip()
            .replace({"": "Sem Marca", "nan": "Sem Marca", "None": "Sem Marca"})
        )

    col_inicial = "Qtd Inicial" if "Qtd Inicial" in df.columns else "Qtd Física"
    if col_inicial not in df.columns:
        df[col_inicial] = 0.0

    data_ref = pd.Timestamp.now().normalize()
    if df_movs is not None and not df_movs.empty and "Data_Emissao" in df_movs.columns:
        datas_validas = pd.to_datetime(df_movs["Data_Emissao"], errors="coerce").dropna()
        if not datas_validas.empty:
            data_ref = datas_validas.max()

    # 1. Demanda Média Diária (DMD)
    df["DMD"] = (df["Total_Saidas"] / max(dias_periodo, 1)).round(2)

    # 2. Estoque Médio e Giro de Estoque
    df["Estoque_Medio"] = ((df[col_inicial] + df["Saldo_Calculado"]) / 2).round(2)
    df["Giro_Estoque"] = np.where(
        df["Estoque_Medio"] > 0,
        (df["Total_Saidas"] / df["Estoque_Medio"]).round(2),
        0.0,
    )

    # 3. Cobertura de Estoque (em dias)
    df["Cobertura_Dias"] = np.where(
        df["DMD"] > 0,
        (df["Saldo_Calculado"].clip(lower=0) / df["DMD"]).round(1),
        np.where(df["Saldo_Calculado"] > 0, 999.0, 0.0),
    )

    # 4. Dias sem Giro (desde a última venda registrada)
    df["Dias_Sem_Giro"] = _calcular_dias_sem_giro(df, data_ref)

    # 5. Preço Médio de Venda, Custo Médio de Compra e Valor Estimado em Estoque
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

    # 6. Curva ABC Financeira (prioriza Valor_Saidas; usa Total_Saidas como fallback)
    usar_financeiro = "Valor_Saidas" in df.columns and df["Valor_Saidas"].sum() > 0
    criterio_abc = "Valor_Saidas" if usar_financeiro else "Total_Saidas"

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

    # 7. Diagnóstico de Status Operacional
    df["Status_Estoque"] = np.select(
        [
            df["Saldo_Calculado"] <= 0,
            df["Cobertura_Dias"] < cobertura_minima,
            (df["Cobertura_Dias"] > dias_excesso) & (df["Cobertura_Dias"] < 999.0),
            df["Cobertura_Dias"] == 999.0,
        ],
        [
            "🔴 Ruptura",
            "🟡 Atenção (Baixo)",
            "🔵 Excesso",
            "⚪ Sem Saída no Período",
        ],
        default="🟢 Saudável",
    )

    # 8. Cálculo de Ruptura e Venda Perdida Estimada
    em_ruptura = df["Saldo_Calculado"] <= 0
    dias_ruptura_bruto = df["Dias_Sem_Giro"].fillna(1.0).clip(lower=1.0)
    df["Dias_Em_Ruptura"] = np.where(
        em_ruptura & (df["DMD"] > 0),
        dias_ruptura_bruto,
        0.0,
    )
    df["Unidades_Perdidas_Est"] = np.where(
        em_ruptura & (df["DMD"] > 0),
        np.ceil(df["DMD"] * df["Dias_Em_Ruptura"]),
        0.0,
    )
    df["Perda_Financeira_Ruptura"] = (
        df["Unidades_Perdidas_Est"] * df["Preco_Medio_Venda"]
    ).round(2)
    df["Perda_Diaria_Ruptura"] = np.where(
        em_ruptura & (df["DMD"] > 0),
        (df["DMD"] * df["Preco_Medio_Venda"]).round(2),
        0.0,
    )

    # 9. Sugestão de Compra para atingir a cobertura alvo (em dias)
    necessidade_bruta = (df["DMD"] * cobertura_alvo) - df["Saldo_Calculado"]
    df["Sugestao_Compra"] = np.where(
        necessidade_bruta > 0,
        np.ceil(necessidade_bruta),
        0.0,
    )

    return df


def _gerar_base_proximo_ciclo(
    df_ind: pd.DataFrame,
    data_registro: str,
    faturamento_total: float,
    valor_estoque_total: float,
    qtd_ruptura: int,
    perda_ruptura_total: float,
) -> pd.DataFrame:
    """Monta a planilha estruturada para continuar a análise no próximo ciclo."""
    colunas_preservar = [
        c for c in ["SKU", "Descrição", "Categoria", "Marca", "Fator_Conversao"] if c in df_ind.columns
    ]
    df_next = df_ind[colunas_preservar].copy()
    df_next["Qtd Inicial"] = df_ind["Saldo_Calculado"].clip(lower=0.0)
    df_next["Saldo_Anterior_Ref"] = df_ind["Saldo_Calculado"]
    df_next["Status_Anterior_Ref"] = df_ind["Status_Estoque"]
    df_next["Ref_Data"] = data_registro
    df_next["Ref_Faturamento"] = round(faturamento_total, 2)
    df_next["Ref_Capital"] = round(valor_estoque_total, 2)
    df_next["Ref_Ruptura_Qtd"] = int(qtd_ruptura)
    df_next["Ref_Perda_Ruptura"] = round(perda_ruptura_total, 2)
    return df_next


def renderizar_aba_indicadores():
    st.header("📊 Indicadores e Simulação de Estoque")
    st.caption(
        "Análise de giro, cobertura, Curva ABC financeira, impacto da ruptura e evolução por ciclos."
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

    # 1. Parâmetros de Simulação
    with st.container(border=True):
        st.subheader("Parâmetros da Simulação")
        p1, p2, p3, p4 = st.columns(4)
        dias_periodo = p1.number_input(
            "Janela analisada (dias)",
            min_value=1,
            max_value=365,
            value=30,
            help="Quantidade de dias coberta pelos XMLs importados.",
        )
        cobertura_minima = p2.number_input(
            "Cobertura mínima / alerta (dias)",
            min_value=1,
            max_value=90,
            value=7,
            help="Abaixo deste limite o item entra em alerta de reposição.",
        )
        cobertura_alvo = p3.number_input(
            "Cobertura alvo de compra (dias)",
            min_value=1,
            max_value=180,
            value=15,
            help="Base de dias para calcular a sugestão de compra.",
        )
        dias_excesso = p4.number_input(
            "Limite para excesso (dias)",
            min_value=15,
            max_value=365,
            value=45,
            help="Acima deste limite o estoque é considerado excessivo.",
        )

    df_ind = calcular_indicadores_estoque(
        df_consolidado=df_consolidado,
        df_movs=df_movs,
        dias_periodo=int(dias_periodo),
        cobertura_minima=int(cobertura_minima),
        cobertura_alvo=int(cobertura_alvo),
        dias_excesso=int(dias_excesso),
    )

    faturamento_total = float(df_ind["Valor_Saidas"].sum())
    valor_estoque_total = float(df_ind["Valor_Estoque_Atual"].sum())
    qtd_ruptura = int((df_ind["Saldo_Calculado"] <= 0).sum())
    perda_ruptura_total = float(df_ind["Perda_Financeira_Ruptura"].sum())
    perda_diaria_total = float(df_ind["Perda_Diaria_Ruptura"].sum())
    qtd_curva_a = int((df_ind["Curva_ABC"] == "A").sum())
    total_sugerido = int(df_ind["Sugestao_Compra"].sum())

    # Verifica se existe um Ponto de Partida salvo para comparação
    snap = st.session_state.get("ponto_partida")
    tem_comparacao = snap is not None

    # 2. Cards de Resumo Executivo (com setas de evolução quando há Ponto de Partida)
    st.subheader("Resumo Executivo")
    if tem_comparacao:
        st.caption(
            f"📈 Comparando ciclo atual contra o **Ponto de Partida** salvo em **{snap['data_registro']}**."
        )

    m1, m2, m3, m4, m5, m6 = st.columns(6)

    delta_fat = None
    delta_cap = None
    delta_rup_qtd = None
    delta_rup_val = None

    if tem_comparacao:
        diff_fat = faturamento_total - float(snap.get("faturamento_total", 0.0))
        diff_cap = valor_estoque_total - float(snap.get("valor_estoque_total", 0.0))
        diff_rup_qtd = qtd_ruptura - int(snap.get("qtd_ruptura", 0))
        diff_rup_val = perda_ruptura_total - float(snap.get("perda_ruptura_total", 0.0))

        delta_fat = f"{_formatar_brl(diff_fat)} vs. P1"
        delta_cap = f"{_formatar_brl(diff_cap)} vs. P1"
        delta_rup_qtd = f"{diff_rup_qtd:+d} SKU(s) vs. P1"
        delta_rup_val = f"{_formatar_brl(diff_rup_val)} vs. P1"

    m1.metric(
        "Faturamento (Saídas)",
        _formatar_brl(faturamento_total),
        delta=delta_fat,
        delta_color="normal",
    )
    m2.metric(
        "Capital em Estoque",
        _formatar_brl(valor_estoque_total),
        delta=delta_cap,
        delta_color="off",
    )
    m3.metric(
        "Itens em Ruptura",
        f"{qtd_ruptura} SKU(s)",
        delta=delta_rup_qtd,
        delta_color="inverse",
    )
    m4.metric(
        "Venda Perdida (Ruptura)",
        _formatar_brl(perda_ruptura_total),
        delta=delta_rup_val
        if tem_comparacao
        else (f"-{_formatar_brl(perda_diaria_total)}/dia" if perda_diaria_total > 0 else None),
        delta_color="inverse",
        help="Estimativa de faturamento deixado na mesa pelos itens zerados com demanda ativa.",
    )
    m5.metric("Curva A (80% Receita)", f"{qtd_curva_a} SKU(s)")
    m6.metric("Sugestão Total Compra", f"{total_sugerido} un.")

    # 3. Controle de Ciclos: Salvar Ponto de Partida
    with st.expander("📌 Ponto de Partida (Salvar Snapshot para Comparar Próximos Ciclos)", expanded=False):
        st.write(
            "Salve o estado atual como **Ponto de Partida (Ponto 1)**. Na próxima simulação, "
            "o saldo final de hoje será usado automaticamente como estoque inicial e os cards mostrarão a evolução."
        )
        c_snap1, c_snap2 = st.columns(2)
        data_agora = datetime.now().strftime("%d/%m/%Y %H:%M")
        df_next_cycle = _gerar_base_proximo_ciclo(
            df_ind=df_ind,
            data_registro=data_agora,
            faturamento_total=faturamento_total,
            valor_estoque_total=valor_estoque_total,
            qtd_ruptura=qtd_ruptura,
            perda_ruptura_total=perda_ruptura_total,
        )

        if c_snap1.button("💾 Salvar Estado Atual como Ponto de Partida", type="secondary", use_container_width=True):
            st.session_state["ponto_partida"] = {
                "data_registro": data_agora,
                "faturamento_total": faturamento_total,
                "valor_estoque_total": valor_estoque_total,
                "qtd_ruptura": qtd_ruptura,
                "perda_ruptura_total": perda_ruptura_total,
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
            help="Guarde este arquivo para continuar a auditoria em outro dia ou computador sem perder o histórico.",
        )

    # 4. Painel Dedicado de Alerta Crítico de Ruptura
    df_criticos = df_ind[
        df_ind["Status_Estoque"].isin(["🔴 Ruptura", "🟡 Atenção (Baixo)"])
    ].sort_values(by=["Perda_Financeira_Ruptura", "Valor_Saidas"], ascending=False)

    if not df_criticos.empty:
        with st.container(border=True):
            st.subheader(
                f"🚨 Alerta Crítico de Ruptura e Risco de Faturamento ({len(df_criticos)} item(ns))"
            )
            st.caption(
                "Produtos zerados ou abaixo da cobertura mínima ordenados pelo maior impacto financeiro."
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
                    "Saldo_Calculado",
                    "DMD",
                    "Dias_Em_Ruptura",
                    "Perda_Financeira_Ruptura",
                    "Perda_Diaria_Ruptura",
                    "Sugestao_Compra",
                ]
                if c in df_criticos.columns
            ]
            st.dataframe(
                df_criticos[cols_ruptura].rename(
                    columns={
                        "Curva_ABC": "Curva",
                        "Status_Estoque": "Situação",
                        "Saldo_Calculado": "Saldo",
                        "Dias_Em_Ruptura": "Dias Zerado",
                        "Perda_Financeira_Ruptura": "Venda Perdida Est. (R$)",
                        "Perda_Diaria_Ruptura": "Perda/Dia (R$)",
                        "Sugestao_Compra": "Comprar Agora (un.)",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            csv_compra_urgente = df_criticos[cols_ruptura].to_csv(index=False, sep=";").encode("utf-8-sig")
            st.download_button(
                label="🛒 Baixar Lista de Compra Urgente / Ruptura (.CSV)",
                data=csv_compra_urgente,
                file_name=f"pedido_reposicao_ruptura_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # 5. Filtros Rápidos e Detalhamento por Produto
    st.subheader("Detalhamento por Produto")
    f1, f2, f3, f4 = st.columns(4)
    filtro_curva = f1.multiselect(
        "Curva ABC:",
        options=["A", "B", "C"],
        default=["A", "B", "C"],
    )
    filtro_status = f2.multiselect(
        "Status de Estoque:",
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
            mapa_saldo_ant = dict(
                zip(
                    df_ref["SKU"].astype(str),
                    pd.to_numeric(df_ref["Saldo_Anterior_Ref"], errors="coerce").fillna(0.0),
                )
            )
            df_filtrado["Var_Saldo"] = df_filtrado["SKU"].astype(str).map(mapa_saldo_ant)
            df_filtrado["Var_Saldo"] = (df_filtrado["Saldo_Calculado"] - df_filtrado["Var_Saldo"]).round(2)

    df_exibicao = df_filtrado.copy()
    df_exibicao["Cobertura_Exibicao"] = np.where(
        df_exibicao["Cobertura_Dias"] >= 999.0,
        "Sem giro (>30d)",
        df_exibicao["Cobertura_Dias"].astype(str) + " d",
    )
    df_exibicao["Dias_Sem_Giro_Exibicao"] = df_exibicao["Dias_Sem_Giro"].apply(
        lambda x: f"{int(x)} d" if pd.notna(x) else "Sem venda"
    )

    colunas_ordenadas = [
        "SKU",
        "Descrição",
        "Categoria",
        "Marca",
        "Curva_ABC",
        "Status_Estoque",
        "Saldo_Calculado",
        "Var_Saldo",
        "Total_Saidas",
        "Valor_Saidas",
        "Perda_Financeira_Ruptura",
        "Participacao_Pct",
        "DMD",
        "Cobertura_Exibicao",
        "Giro_Estoque",
        "Dias_Sem_Giro_Exibicao",
        "Ultima_Saida",
        "Sugestao_Compra",
    ]
    colunas_presentes = [c for c in colunas_ordenadas if c in df_exibicao.columns]

    st.dataframe(
        df_exibicao[colunas_presentes].rename(
            columns={
                "Curva_ABC": "Curva ABC",
                "Status_Estoque": "Status",
                "Saldo_Calculado": "Saldo Atual",
                "Var_Saldo": "Var. Saldo (vs P1)",
                "Total_Saidas": "Saídas (Qtd)",
                "Valor_Saidas": "Faturamento (R$)",
                "Perda_Financeira_Ruptura": "Venda Perdida (R$)",
                "Participacao_Pct": "% Receita",
                "Cobertura_Exibicao": "Cobertura",
                "Giro_Estoque": "Giro",
                "Dias_Sem_Giro_Exibicao": "Dias s/ Giro",
                "Ultima_Saida": "Última Venda",
                "Sugestao_Compra": "Sugestão Compra",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # 6. Evolução Diária de Saídas
    if df_movs is not None and not df_movs.empty and "Data_Emissao" in df_movs.columns:
        df_saidas_dia = df_movs[
            (df_movs["Tipo"] == "Saída") & (df_movs["Data_Emissao"] != "")
        ].copy()
        if not df_saidas_dia.empty:
            st.subheader("Evolução Diária de Vendas (Saídas)")
            resumo_diario = (
                df_saidas_dia.groupby("Data_Emissao", as_index=False)["Quantidade"]
                .sum()
                .sort_values("Data_Emissao")
            )
            st.bar_chart(resumo_diario.set_index("Data_Emissao")["Quantidade"])

    # 7. Exportação do Relatório Completo de Indicadores
    data_hoje = datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_indicadores = df_filtrado.to_csv(index=False, sep=";").encode("utf-8-sig")
    st.download_button(
        label="📥 Baixar Relatório Completo de Indicadores (.CSV)",
        data=csv_indicadores,
        file_name=f"indicadores_estoque_{data_hoje}.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )
