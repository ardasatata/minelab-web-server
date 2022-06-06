import * as React from 'react';
import styled from 'styled-components/macro';
import ncuWhite from '../../../assets/ncu-white.png';
import minelab from '../../../assets/minelab.png';
import { ReactComponent as NTUAicon } from '../../../assets/ntua-white.svg';

export function Logo() {
  return (
    <Wrapper>
      <img src={ncuWhite} className={'h-10 mt-1'} />
      <NTUAicon className={'h-12 whitespace-nowrap block'} style={{}} />
        <div className={"flex flex-col items-center justify-center text-center"}>
            <Title style={{ whiteSpace: 'normal', fontSize: 18 }}>二胡基礎學習診斷</Title>
            <Title style={{ whiteSpace: 'normal', fontSize: 14 }}>Fundamental Erhu Learning Diagnosis</Title>
        </div>
      {/*<Description>Minelab 2022</Description>*/}
    </Wrapper>
  );
}

const Wrapper = styled.div`
  display: flex;
  align-items: center;
`;

const Title = styled.div`
  font-size: 1.25rem;
  color: ${p => p.theme.text};
  font-weight: bold;
  margin-right: 1rem;
  margin-left: 1rem;
`;

const Description = styled.div`
  font-size: 0.875rem;
  color: ${p => p.theme.textSecondary};
  font-weight: normal;
`;
