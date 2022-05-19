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
      <Title style={{ whiteSpace: 'normal' }}>Erhu Trainer | 二胡學習</Title>
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
